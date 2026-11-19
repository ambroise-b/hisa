"""HiSA in plain PyTorch: the readable definition, used as the reference in the tests.

    s_ij = 1/sqrt(d) * sum_p 1(0 <= i - j < w_p) * <q_i[p], k_j[p]>,   w_p = Phi / theta_p,

s_ij = -inf where no pair is kept (or j > i); fp32 softmax; P V. With rule_block = S, the block rule of
Flash-HiSA: i - j is replaced by the minimum distance between the S-token blocks of i and j.

It materialises a [block, T, P] tensor per head: fine for tests and short sequences, not for training.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from hisa import rope


def sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Dense causal attention (Phi = inf)."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=q.shape[1] != k.shape[1])


def _split_pairs(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    P = x.shape[-1] // 2
    return x[..., :P], x[..., P:]


def _group(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """q [B, H, T, d] -> [B, Hkv, G, T, d]; k, v -> [B, Hkv, 1, T, d]."""
    B, H, T, d = q.shape
    Hkv = k.shape[1]
    return q.view(B, Hkv, H // Hkv, T, d), k.unsqueeze(2), v.unsqueeze(2)


def hisa_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    theta: torch.Tensor,
    max_angle_deg: float = 90.0,
    rule_block: int | None = None,
    block_size: int = 64,
) -> torch.Tensor:
    """q [B, H, T, d], k, v [B, H_kv, T, d] (rotated, HF pair layout; GQA: query head h uses kv head
    h // (H / H_kv)) -> [B, H, T, d]. theta: the RoPE frequencies of the rule ([d / 2]).
    rule_block: None = exact rule, S = block rule. block_size: queries processed at a time (memory only)."""
    B, H, T, d = q.shape
    qg, kg, vg = _group(q, k, v)
    q1, q2 = _split_pairs(qg)
    k1, k2 = _split_pairs(kg)
    masks = rope.pair_masks(theta, max_angle_deg, T, device=q.device, rule_block=rule_block)  # [P, T, T]
    any_active = masks.any(0)  # [T, T]
    acc = torch.promote_types(q.dtype, torch.float32)  # fp32 scores (fp64 for fp64 inputs)
    outs = []
    for i0 in range(0, T, block_size):
        i1 = min(T, i0 + block_size)
        # c[..., i, j, p] = <q_i[p], k_j[p]>
        c = q1[..., i0:i1, None, :].to(acc) * k1[..., None, :, :].to(acc)
        c = c + q2[..., i0:i1, None, :].to(acc) * k2[..., None, :, :].to(acc)
        m = masks[:, i0:i1, :].permute(1, 2, 0)  # [bq, T, P]
        s = (c * m).sum(-1) / math.sqrt(d)
        s = s.masked_fill(~any_active[i0:i1], -math.inf)
        probs = torch.softmax(s, dim=-1)
        outs.append(torch.matmul(probs.to(vg.dtype), vg))
    return torch.cat(outs, dim=-2).reshape(B, H, T, d)
