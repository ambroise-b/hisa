"""Flash-HiSA: the block rule inside FlashAttention-3 (Hopper, sm90), forward and backward.

The kernels live in the HiSA fork of FlashAttention (third_party/flash-attention, branch `hisa`). Build it with
scripts/build_flash.sh (in place) or `pip install` its hopper/ directory; see docs/install.md. Only head
dimension 64 and bf16 are supported (other dtypes are cast to bf16 and the output cast back).

The rule (block size S = rule_block, default 128): query block bi and key block bj keep pair p iff
dmin(bi, bj) = max((bi - bj) * S - (S - 1), 0) < L_p. The kernel receives it as a small int32 table: for each
block distance nd, the number a[nd] of dropped pairs (a prefix of the pairs, since the windows grow with p).
Chunks of 16 features (8 pairs) below a[nd] // 8 are skipped in Q K^T; the dropped pairs of the boundary chunk
are zeroed in shared-memory Q. q and k are used in the chunked feature layout (hisa.rope.chunk_permutation);
HF-layout inputs are permuted here, and autograd maps the gradients back.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

# Built in place by scripts/build_flash.sh: <repo>/third_party/flash-attention/hopper. HISA_FLASH_PATH overrides it
# (the hopper/ directory of another in-place build of the fork).
HOPPER = Path(os.environ.get("HISA_FLASH_PATH") or
              Path(__file__).resolve().parents[2] / "third_party" / "flash-attention" / "hopper")


_CHECKED = False  # the op schema was checked once (interface() runs at every call)


def interface():
    """The FlashAttention-3 Python interface of the HiSA fork (registers torch.ops.flash_attn_3)."""
    try:
        import flash_attn_interface
    except ImportError:
        if str(HOPPER) not in sys.path:  # hopper/flash_attn_interface.py next to the in-place build
            sys.path.insert(0, str(HOPPER))
        try:
            import flash_attn_interface
        except ImportError as e:
            raise ImportError("Flash-HiSA needs the HiSA FlashAttention-3 fork (third_party/flash-attention): "
                              "build it with scripts/build_flash.sh, see docs/install.md") from e
    global _CHECKED
    if not _CHECKED and "sparse_a" not in str(torch.ops.flash_attn_3.fwd.default._schema):
        raise ImportError("the installed FlashAttention-3 is not the HiSA fork (no sparse_a argument): "
                          "build third_party/flash-attention, branch hisa")
    _CHECKED = True
    return flash_attn_interface


def flash_dense_attention(q, k, v):
    """Dense causal FlashAttention-3 from the same build (the baseline): q [B, H, T, d], k, v [B, H_kv, T, d]."""
    fa = interface()
    dt = torch.get_autocast_dtype(q.device.type) if torch.is_autocast_enabled(q.device.type) else q.dtype
    bf = torch.bfloat16
    out = fa.flash_attn_func(q.to(bf).transpose(1, 2), k.to(bf).transpose(1, 2), v.to(bf).transpose(1, 2),
                             causal=True)
    out = out[0] if isinstance(out, tuple) else out
    return out.transpose(1, 2).to(dt)


# --------------------------------------------------------------------------- the block-rule table


def sparse_table(L: list[int], rule_block: int, T: int) -> list[int]:
    """a[nd] = number of pairs dropped between rule blocks nd apart (dmin = max(nd * S - S + 1, 0)), for the
    distances where at least one pair is kept (nd < len). Pairs are dropped iff L_p <= dmin (L increasing)."""
    out = []
    for nd in range(-(-T // rule_block) + 1):
        dmin = max(nd * rule_block - rule_block + 1, 0)
        a = sum(1 for l in L if l <= dmin)
        if a >= len(L):
            break
        out.append(a)
    return out


def table_with_header(table: list[int], L: list[int] | None = None, rule_block: int = 128) -> list[int]:
    """[nd_1, .., nd_4] + [L_0, .., L_31] + table: nd_k = first distance with a >= 8 k (len(table) if none), from
    which the kernel derives the number of skipped 16-feature chunks; L = the integer thresholds, from which it
    computes a inside a segment without memory loads. L = None: thresholds consistent with the (monotone) table,
    pair p dropped from the first distance where table > p (for hand-made tables in tests / benchmarks)."""
    n = len(table)
    nd = [next((i for i, a in enumerate(table) if a >= 8 * k), n) for k in (1, 2, 3)]
    dmin = lambda d: max(d * rule_block - rule_block + 1, 0)
    if L is None:
        L = [dmin(next((i for i, a in enumerate(table) if a > p), n)) for p in range(32)]
    return nd + [n] + [min(int(x), 1 << 30) for x in L] + list(table)


_TABLE_CACHE: dict = {}


def _table_from_list(L: list[int], rule_block: int, T: int, device):
    """(int32 table with header on the device, number of block distances with a kept pair), cached by the
    threshold values (plain ints: no GPU sync, constant under torch.compile)."""
    key = (tuple(L), rule_block, T, str(device))
    if key not in _TABLE_CACHE:
        table = sparse_table(list(L), rule_block, T)
        _TABLE_CACHE[key] = (torch.tensor(table_with_header(table, list(L), rule_block), dtype=torch.int32,
                                          device=device), len(table))
    return _TABLE_CACHE[key]


def _window_arg(n: int, rule_block: int, T: int) -> int:
    window = n * rule_block - 1  # every kept pair has i - j <= n * S - 1
    return window if window < T - 1 else -1


_PERM: dict = {}


def _perm(device) -> torch.Tensor:
    if device not in _PERM:
        from hisa.rope import chunk_permutation

        _PERM[device] = chunk_permutation().to(device)
    return _PERM[device]


def flash_block_sparse_fwd_bthd(qc, kc, vc, table, window: int, rule_block: int = 128):
    """Kernel call only: qc, kc [B, T, H(_kv), 64] in the chunked feature layout, vc [B, T, H_kv, 64] (unit stride
    on d); table = table_with_header(sparse_table(..), L) as int32 on the device;
    window = len(sparse_table) * rule_block - 1. Returns (out [B, T, H, 64], lse [B, H, T], natural log)."""
    T = qc.shape[1]
    out, lse, _, _ = torch.ops.flash_attn_3.fwd(
        qc, kc, vc, is_causal=True, window_size_left=window if window < T - 1 else -1, window_size_right=0,
        num_splits=1, pack_gqa=False, sparse_a=table, sparse_rule_block=rule_block)
    return out, lse


# --------------------------------------------------------------------------- custom ops (autograd, torch.compile)
# Opaque to torch.compile; the fake kernels give the output shapes. q / k chunked, all [B, T, H, d] views.


@torch.library.custom_op("hisa::flash_fwd", mutates_args=())
def flash_fwd(qb: torch.Tensor, kb: torch.Tensor, vb: torch.Tensor, L: list[int], rule_block: int
              ) -> tuple[torch.Tensor, torch.Tensor]:
    interface()
    table, n = _table_from_list(L, rule_block, qb.shape[1], qb.device)
    return flash_block_sparse_fwd_bthd(qb, kb, vb, table, n * rule_block - 1, rule_block)


@flash_fwd.register_fake
def _(qb, kb, vb, L, rule_block):
    B, T, H, d = qb.shape
    return qb.new_empty(B, T, H, d), qb.new_empty(B, H, T, dtype=torch.float32)


@torch.library.custom_op("hisa::flash_bwd", mutates_args=())
def flash_bwd(dob: torch.Tensor, qb: torch.Tensor, kb: torch.Tensor, vb: torch.Tensor, out: torch.Tensor,
              lse: torch.Tensor, L: list[int], rule_block: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    interface()
    T = qb.shape[1]
    table, n = _table_from_list(L, rule_block, T, qb.device)
    if dob.stride(-1) != 1:
        dob = dob.contiguous()
    dq, dk, dv = qb.new_empty(qb.shape), kb.new_empty(kb.shape), vb.new_empty(vb.shape)
    torch.ops.flash_attn_3.bwd(dob, qb, kb, vb, out, lse, dq, dk, dv, is_causal=True,
                               window_size_left=_window_arg(n, rule_block, T), window_size_right=0,
                               sparse_a=table, sparse_rule_block=rule_block)
    return dq, dk, dv


@flash_bwd.register_fake
def _(dob, qb, kb, vb, out, lse, L, rule_block):
    return qb.new_empty(qb.shape), kb.new_empty(kb.shape), vb.new_empty(vb.shape)


def _setup_context(ctx, inputs, output):
    qb, kb, vb, L, rule_block = inputs
    out, lse = output
    ctx.save_for_backward(qb, kb, vb, out, lse)
    ctx.L, ctx.rule_block = L, rule_block


def _backward(ctx, dout, dlse):
    qb, kb, vb, out, lse = ctx.saved_tensors
    dq, dk, dv = flash_bwd(dout, qb, kb, vb, out, lse, ctx.L, ctx.rule_block)
    return dq, dk, dv, None, None


flash_fwd.register_autograd(_backward, setup_context=_setup_context)


def hisa_flash(q, k, v, L, rule_block: int = 128, chunked: bool = False):
    """Flash-HiSA attention, forward and backward (bf16; other dtypes are cast, and under autocast the output has
    the autocast dtype). q [B, H, T, 64], k, v [B, H_kv, T, 64] (rotated); L = hisa.rope.thresholds(...) as a list
    of ints (a tensor is converted, which syncs). chunked=True: q / k are already in the chunked feature layout
    (e.g. RoPE applied with hisa.rope.apply_rotary_chunked): no permutation copy. q / k / v are passed to the
    kernel as [B, T, H, d] views, so [B, H, T, d] views of [B, T, H, d] projections are not copied."""
    if isinstance(L, torch.Tensor):
        L = L.tolist()
    dt = torch.get_autocast_dtype(q.device.type) if torch.is_autocast_enabled(q.device.type) else q.dtype
    bf = torch.bfloat16
    q, k, v = q.to(bf), k.to(bf), v.to(bf)
    if not chunked:
        perm = _perm(q.device)
        q, k = q.index_select(-1, perm), k.index_select(-1, perm)
    out, _ = flash_fwd(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), list(L), rule_block)
    return out.transpose(1, 2).to(dt)
