"""Flash-HiSA decoding: block-rule attention of one new token per sequence over a ring-buffer KV cache.

A cached key needs pair p only while it is within the pair's window, and its value only within the reach. The
cache therefore stores the keys as 4 ring buffers, one per 16-feature chunk (8 RoPE pairs) of the chunked layout,
each as long as the window of its slowest pair rounded up to whole 128-token rule blocks, and the values in one
ring as long as the reach. The token at position t lives in slot t mod M_c of ring c (length M_c); since every ring
holds whole blocks, a 128-key tile never wraps. The decode kernel (FlashAttention-3 fork, sparse_kv.h) reads, for
each key block, only the chunks that are still kept, with cp.async; the others are never read.

Shapes follow FlashAttention-3's [batch, seq, head, dim]: K ring c is [B, M_c, H_kv, 16], V is [B, M_V, H_kv, 64].
With max_angle_deg = inf every ring has S_max slots and no chunk is dropped: the same kernel, dense (the
apples-to-apples baseline). All sequences of the batch decode the same position t (teacher forcing / batched
generation of equal-length prompts).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from hisa import rope
from hisa.kernels.flash import _table_from_list, interface

RULE_BLOCK = 128


def blocks_kept(LC: int, rule_block: int, S_max: int) -> int:
    """Number of block distances nd >= 0 with dmin(nd) = max(nd * S - (S - 1), 0) < LC (capped)."""
    n, cap = 1, -(-S_max // rule_block) + 1
    while n < cap and n * rule_block - (rule_block - 1) < LC:
        n += 1
    return n


def ring_lengths(L: list[int], S_max: int, rule_block: int = RULE_BLOCK) -> tuple[list[int], int]:
    """(key ring length M_c of each 16-feature chunk, value ring length M_V) for thresholds L."""
    lc = [max(L[8 * c:8 * c + 8]) for c in range(4)]
    M = [min(blocks_kept(x, rule_block, S_max) * rule_block, S_max) for x in lc]
    return M, max(M)


@triton.jit
def _append_kernel(Kn, Vn, K0, K1, K2, K3, V, Tptr, Hkv, R0, R1, R2, R3, RV):
    """New key (chunked layout) and value of every (sequence, KV head) -> slot t % M of each ring ([B, M, H_kv, d])."""
    pid = tl.program_id(0).to(tl.int64)
    b = pid // Hkv
    hk = pid % Hkv
    t = tl.load(Tptr)
    f16 = tl.arange(0, 16)
    f64 = tl.arange(0, 64)
    src = pid * 64
    tl.store(K0 + ((b * R0 + t % R0) * Hkv + hk) * 16 + f16, tl.load(Kn + src + f16))
    tl.store(K1 + ((b * R1 + t % R1) * Hkv + hk) * 16 + f16, tl.load(Kn + src + 16 + f16))
    tl.store(K2 + ((b * R2 + t % R2) * Hkv + hk) * 16 + f16, tl.load(Kn + src + 32 + f16))
    tl.store(K3 + ((b * R3 + t % R3) * Hkv + hk) * 16 + f16, tl.load(Kn + src + 48 + f16))
    tl.store(V + ((b * RV + t % RV) * Hkv + hk) * 64 + f64, tl.load(Vn + src + f64))


class HiSAKVCache:
    """Ring-buffer KV cache of all layers for the Flash-HiSA decode kernel (block rule, S = 128, bf16, d = 64).

    theta: the RoPE frequencies of the rule ([32]); S_max: maximum sequence length (a multiple of 128).
    Use: for each layer, append(layer, k_t, v_t, t) then attend(layer, q_t, t), with q / k in the chunked layout
    (hisa.rope.chunk_permutation; `to_chunked` converts from the HF layout) and t a device int32 tensor [1]."""

    def __init__(self, n_layers: int, B: int, H: int, Hkv: int, S_max: int, theta: torch.Tensor,
                 max_angle_deg: float = 90.0, device="cuda", dtype=torch.bfloat16, num_splits: int = 0):
        if S_max % RULE_BLOCK:
            raise ValueError("S_max must be a multiple of 128 (whole key tiles)")
        if dtype != torch.bfloat16:
            raise ValueError("the Flash-HiSA decode kernel is bf16 only")
        device = torch.device(device)
        self.B, self.H, self.Hkv, self.S_max, self.num_splits = B, H, Hkv, S_max, num_splits
        self.L = [int(x) for x in rope.thresholds(theta.cpu(), max_angle_deg).tolist()]
        self.R, self.RV = ring_lengths(self.L, S_max)
        self.k = [torch.zeros(n_layers, B, r, Hkv, 16, device=device, dtype=dtype) for r in self.R]
        self.v = torch.zeros(n_layers, B, self.RV, Hkv, 64, device=device, dtype=dtype)
        self.table, n = _table_from_list(self.L, RULE_BLOCK, S_max, device)
        self.window = n * RULE_BLOCK - 1  # FlashAttention turns it off (-1) when it covers the whole cache
        self.seqlens = torch.zeros(B, device=device, dtype=torch.int32)
        self._perm = rope.chunk_permutation().to(device)
        interface()

    @property
    def nbytes(self) -> int:
        """Memory of the cache (all layers)."""
        return sum(x.numel() * x.element_size() for x in [*self.k, self.v])

    def to_chunked(self, x: torch.Tensor) -> torch.Tensor:
        """HF feature layout -> chunked layout (last dim)."""
        return x.index_select(-1, self._perm)

    def append(self, layer: int, k_chunked: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> None:
        """k_chunked, v [B, H_kv, 64] of position t (device int32 [1])."""
        _append_kernel[(self.B * self.Hkv,)](k_chunked.contiguous(), v.contiguous(), *[x[layer] for x in self.k],
                                             self.v[layer], t, self.Hkv, *self.R, self.RV)

    def attend(self, layer: int, q_chunked: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """q_chunked [B, H, 64] of position t -> attention output [B, H, 64] over positions 0 .. t."""
        self.seqlens.copy_((t + 1).expand(self.B))
        v = self.v[layer]
        out, _, _, _ = torch.ops.flash_attn_3.fwd(
            q_chunked[:, None], v, v, seqused_k=self.seqlens, is_causal=False,
            window_size_left=self.window if self.window < self.RV - 1 else -1, window_size_right=-1,
            num_splits=self.num_splits, sparse_a=self.table, sparse_rule_block=RULE_BLOCK,
            sparse_kc0=self.k[0][layer], sparse_kc1=self.k[1][layer], sparse_kc2=self.k[2][layer],
            sparse_kc3=self.k[3][layer])
        return out[:, 0]
