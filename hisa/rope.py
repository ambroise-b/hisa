"""RoPE quantities of the HiSA rule.

Hugging Face (Llama) layout: frequency pair p is made of features (p, p + P), P = head_dim / 2, with frequency
theta_p = base^(-2p / d). HiSA keeps the score part of pair p between query i and key j only while the rotation
(i - j) * theta_p stays below the angle budget Phi, i.e. while i - j < w_p = Phi / theta_p (the pair's window).
"""

from __future__ import annotations

import math

import torch

BIG = 1 << 30  # integer threshold for "always kept" (infinite budget = dense attention)


def inv_freq(head_dim: int = 64, base: float = 10000.0) -> torch.Tensor:
    """theta_p, identical to Hugging Face's default RoPE frequencies (float32, [head_dim / 2])."""
    return 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))


def max_angle_rad(max_angle_deg: float) -> float:
    return math.radians(max_angle_deg) if math.isfinite(max_angle_deg) else math.inf


def window(theta: torch.Tensor, max_angle_deg: float) -> torch.Tensor:
    """w_p = Phi / theta_p (float64, [P]): pair p is kept while 0 <= i - j < w_p."""
    phi = max_angle_rad(max_angle_deg)
    theta = theta.double()
    if math.isinf(phi):
        return torch.full_like(theta, math.inf)
    return phi / theta


def reach(theta: torch.Tensor, max_angle_deg: float) -> float:
    """Distance beyond which no pair is kept (keys further away are outside the softmax)."""
    return float(window(theta, max_angle_deg).max())


def thresholds(theta: torch.Tensor, max_angle_deg: float, device=None) -> torch.Tensor:
    """Integer windows L_p = ceil(w_p) (int32 [P], capped at BIG): for an integer distance D,
    D < L_p  <=>  D < w_p. Computed in float64 on the host, so every kernel applies exactly the same rule."""
    w = window(theta, max_angle_deg)
    L = torch.where(torch.isfinite(w), torch.ceil(w), torch.full_like(w, BIG)).clamp(max=BIG)
    return L.to(torch.int32).to(device)


def pair_masks(theta: torch.Tensor, max_angle_deg: float, T: int, device=None,
               rule_block: int | None = None) -> torch.Tensor:
    """M[p, i, j] = 1(0 <= i - j < w_p), bool [P, T, T] (causal).

    rule_block = S (block rule): the distance is replaced by the minimum distance between the blocks of i and j,
    max((i // S - j // S) * S - (S - 1), 0): M[p, i, j] = 1(j <= i, dmin < w_p)."""
    w = window(theta, max_angle_deg).to(device)
    pos = torch.arange(T, device=device)
    delta = pos[:, None] - pos[None, :]
    dist = delta
    if rule_block is not None:
        blk = pos // rule_block
        dist = ((blk[:, None] - blk[None, :]) * rule_block - (rule_block - 1)).clamp(min=0)
    return (delta >= 0) & (dist.double() < w[:, None, None])


# --------------------------------------------------------------------------- chunked feature layout


def chunk_permutation(head_dim: int = 64, chunk: int = 16) -> torch.Tensor:
    """perm with x_chunked = x[..., perm]: chunk c (features [chunk c, chunk (c + 1))) holds pairs
    chunk / 2 * c .. chunk / 2 * (c + 1) - 1, first the features p then the features p + P. The kernels work in
    this layout, so that skipping the fastest pairs means skipping whole 16-feature chunks."""
    P, half = head_dim // 2, chunk // 2
    return torch.tensor([(c * half + j) if j < half else (P + c * half + j - half)
                         for c in range(head_dim // chunk) for j in range(chunk)])


def rotate_half_chunked(x: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    """Hugging Face rotate_half ([a, b] -> [-b, a]) applied inside every chunk of the chunked layout."""
    x4 = x.unflatten(-1, (x.shape[-1] // chunk, 2, chunk // 2))
    return torch.stack((-x4[..., 1, :], x4[..., 0, :]), dim=-2).flatten(-3)


def apply_rotary_chunked(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """RoPE with q / k in the chunked layout ([B, H, T, d]); cos / sin as returned by Hugging Face's rotary
    embedding ([B, T, d], HF layout). Same function as HF's apply_rotary_pos_emb on x[..., perm]."""
    perm = chunk_permutation(q.shape[-1]).to(cos.device)
    cos, sin = cos[..., perm].unsqueeze(1), sin[..., perm].unsqueeze(1)
    return q * cos + rotate_half_chunked(q) * sin, k * cos + rotate_half_chunked(k) * sin
