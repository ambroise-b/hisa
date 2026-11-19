"""HiSA, block rule: Triton kernels (forward + backward), the portable version of Flash-HiSA's rule.

The block rule is a different (slightly denser) model than the exact rule. Positions are grouped into rule blocks
of S tokens (rule_block, default 128). For query i in block bi = i // S and key j <= i in block bj = j // S, the
rule uses the minimum distance between the two blocks,

    dmin = max((bi - bj) * S - (S - 1), 0),

and keeps pair p on the whole block pair iff dmin < L_p (L_p = ceil(w_p), as in the exact rule):

    s_ij = 1/sqrt(d) * sum_{p : dmin(bi, bj) < L_p} <q_i[p], k_j[p]>,   s_ij = -inf if no pair is kept.

Since dmin <= i - j, every pair kept by the exact rule is kept; pairs are kept up to S - 1 tokens beyond their
exact cutoff. The rule depends on the block grid (absolute positions mod S), so S is part of the model: training
and inference must use the same S. Kernel tiles divide S, so the kept pairs are uniform on each kernel tile: there
are no per-element pair masks. Reference: hisa.reference.hisa_attention_reference(..., rule_block=S).

Layout and compute as in triton_exact: q and k are permuted so that chunk c (16 contiguous features) holds pairs
8c .. 8c+7. With a = #{p : L_p <= dmin} pairs dropped on a tile, chunks c >= a // 8 are computed by 16-wide
tensor-core dots (only those slices of k / q are loaded), and in chunk a // 8 the columns of pairs < a are zeroed.
The key loop of a query tile runs in 4 segments (far -> near) with 1, 2, 3 and 4 computed chunks; the segment
boundaries are multiples of S. P V, dV and dP are dense.

Backward: by default one fused kernel (one program per key tile, looping over the query tiles that see it):
S, P, dP and dS are computed once per tile pair; dK and dV accumulate in registers and dQ = dS K is added with
fp32 atomics into a zeroed buffer (run-to-run nondeterministic in the last bits, like FlashAttention-2).
fused_bwd=False runs the two-kernel backward (dQ kernel + dK/dV kernel, deterministic, scores computed twice).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from hisa.kernels.triton_exact import (  # noqa: F401  (jit helpers are resolved as module globals)
    HEAD_DIM,
    LOG2E,
    _contiguous_like,
    _count_le,
    _keep,
    _ld,
    _perm,
    _prec,
    thresholds,
)

# --------------------------------------------------------------------------- kernel helpers


@triton.jit
def _dmin(i0, j0, RB):
    """Minimum distance between the rule blocks of query i0 and key j0."""
    return tl.maximum((i0 // RB - j0 // RB) * RB - RB + 1, 0)


@triton.jit
def _nblk(L, RB):
    """Smallest block difference n with dmin(n) = (n - 1) * RB + 1 >= L (L >= 1)."""
    return (L + 2 * RB - 2) // RB


@triton.jit
def _qk(x, y, s, c: tl.constexpr, C0: tl.constexpr, a, PREC: tl.constexpr):
    """s += keep(x) @ y^T for chunk c; only the lowest computed chunk (C0) has dropped pairs to zero."""
    if c == C0:
        x = _keep(x, c, a)
    return tl.dot(x, tl.trans(y), s, input_precision=PREC)


@triton.jit
def _acc(acc, ds, x, c: tl.constexpr, C0: tl.constexpr, a, PREC: tl.constexpr):
    """acc += ds @ keep(x) for chunk c (gradient of chunk c of q or k)."""
    if c == C0:
        x = _keep(x, c, a)
    return tl.dot(ds, x, acc, input_precision=PREC)


@triton.jit
def _scores(q0, q1, q2, q3, K, kv_base, offs_n, kmask, a, C0: tl.constexpr, D: tl.constexpr, PREC: tl.constexpr):
    """Unscaled scores of resident query chunks against the key tile (chunks >= C0 only)."""
    s = tl.zeros([q0.shape[0], offs_n.shape[0]], tl.float32)
    if C0 <= 0:
        s = _qk(q0, _ld(K, kv_base, offs_n, kmask, 0, D), s, 0, C0, a, PREC)
    if C0 <= 1:
        s = _qk(q1, _ld(K, kv_base, offs_n, kmask, 1, D), s, 1, C0, a, PREC)
    if C0 <= 2:
        s = _qk(q2, _ld(K, kv_base, offs_n, kmask, 2, D), s, 2, C0, a, PREC)
    s = _qk(q3, _ld(K, kv_base, offs_n, kmask, 3, D), s, 3, C0, a, PREC)
    return s


@triton.jit
def _key_segments(Lptr, i_lo, j_end, RB):
    """For a query tile: first key, and the key starts where 3 / 2 / 1 chunks become 2 / 1 / 0 dropped."""
    bi = i_lo // RB
    j_start = tl.maximum(bi - _nblk(tl.load(Lptr + 31), RB) + 1, 0) * RB  # blocks with at least one pair
    b3 = tl.minimum(tl.maximum((bi - _nblk(tl.load(Lptr + 23), RB) + 1) * RB, j_start), j_end)
    b2 = tl.minimum(tl.maximum((bi - _nblk(tl.load(Lptr + 15), RB) + 1) * RB, j_start), j_end)
    b1 = tl.minimum(tl.maximum((bi - _nblk(tl.load(Lptr + 7), RB) + 1) * RB, j_start), j_end)
    return j_start, b3, b2, b1


@triton.jit
def _query_segments(Lptr, j_lo, i_start, T, RB):
    """For a key tile: query ends of the 0 / 1 / 2 / 3 dropped-chunk segments (near -> far), and the last query."""
    bj = j_lo // RB
    i_end = tl.minimum((bj + _nblk(tl.load(Lptr + 31), RB)) * RB, T)
    s1 = tl.minimum(tl.maximum((bj + _nblk(tl.load(Lptr + 7), RB)) * RB, i_start), i_end)
    s2 = tl.minimum(tl.maximum((bj + _nblk(tl.load(Lptr + 15), RB)) * RB, i_start), i_end)
    s3 = tl.minimum(tl.maximum((bj + _nblk(tl.load(Lptr + 23), RB)) * RB, i_start), i_end)
    return s1, s2, s3, i_end


def _configs(kind: str):
    if os.environ.get("TRITON_INTERPRET") == "1":
        return [triton.Config({"BM": 32, "BN": 32}, num_warps=4, num_stages=1)]
    shapes = [(128, 64, 8, 3), (128, 64, 4, 3), (64, 64, 4, 3), (128, 128, 8, 2), (64, 32, 4, 3), (32, 32, 4, 3)]
    if kind == "dkdv":
        shapes = [(64, 128, 8, 3), (64, 64, 4, 3), (32, 64, 4, 3), (128, 128, 8, 2), (32, 128, 8, 3), (32, 32, 4, 3)]
    return [triton.Config({"BM": m, "BN": n}, num_warps=w, num_stages=st) for m, n, w, st in shapes]


def _prune(configs, named_args, **kwargs):
    """Kernel tiles must divide the rule block (the kept pairs are then uniform on a tile)."""
    RB = named_args.get("RB", kwargs.get("RB"))
    return [c for c in configs if RB % c.kwargs["BM"] == 0 and RB % c.kwargs["BN"] == 0]


# --------------------------------------------------------------------------- forward


@triton.jit
def _fwd_tile(m_i, l_i, acc, q0, q1, q2, q3, K, V, kv_base, j0, offs_m, i_lo, T, Lv, qk_scale, RB,
              C0: tl.constexpr, D: tl.constexpr, BN: tl.constexpr, PREC: tl.constexpr):
    offs_n = j0 + tl.arange(0, BN)
    kmask = offs_n < T
    a = _count_le(Lv, _dmin(i_lo, j0, RB))
    s = _scores(q0, q1, q2, q3, K, kv_base, offs_n, kmask, a, C0, D, PREC) * qk_scale
    if (j0 + BN - 1 > i_lo) | (j0 + BN > T):
        s = tl.where((offs_m[:, None] >= offs_n[None, :]) & kmask[None, :], s, float("-inf"))
    v = tl.load(V + kv_base + offs_n[:, None] * D + tl.arange(0, D)[None, :], mask=kmask[:, None], other=0.0)
    m_new = tl.maximum(m_i, tl.max(s, axis=1))
    m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    p = tl.exp2(s - m_safe[:, None])
    alpha = tl.exp2(m_i - m_safe)
    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision=PREC)
    return m_new, l_i, acc


@triton.autotune(configs=_configs("fwd"), key=["T", "H", "G", "RB"], prune_configs_by={"early_config_prune": _prune})
@triton.jit
def _fwd_kernel(Q, K, V, O, LSE, Lptr, qk_scale, T, H, G, RB,
                D: tl.constexpr, P: tl.constexpr, PREC: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    hk = (bh % H) // G
    q_base = bh.to(tl.int64) * T * D
    kv_base = (b * (H // G) + hk).to(tl.int64) * T * D
    offs_m = pid_m * BM + tl.arange(0, BM)
    rmask = offs_m < T
    q0 = _ld(Q, q_base, offs_m, rmask, 0, D)
    q1 = _ld(Q, q_base, offs_m, rmask, 1, D)
    q2 = _ld(Q, q_base, offs_m, rmask, 2, D)
    q3 = _ld(Q, q_base, offs_m, rmask, 3, D)
    Lv = tl.load(Lptr + tl.arange(0, P))

    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    i_lo = pid_m * BM
    j_end = tl.minimum(i_lo + BM - 1, T - 1) + 1
    j_start, b3, b2, b1 = _key_segments(Lptr, i_lo, j_end, RB)
    for j0 in range(j_start, b3, BN):
        m_i, l_i, acc = _fwd_tile(m_i, l_i, acc, q0, q1, q2, q3, K, V, kv_base, j0, offs_m, i_lo, T, Lv, qk_scale,
                                  RB, 3, D, BN, PREC)
    for j0 in range(b3, b2, BN):
        m_i, l_i, acc = _fwd_tile(m_i, l_i, acc, q0, q1, q2, q3, K, V, kv_base, j0, offs_m, i_lo, T, Lv, qk_scale,
                                  RB, 2, D, BN, PREC)
    for j0 in range(b2, b1, BN):
        m_i, l_i, acc = _fwd_tile(m_i, l_i, acc, q0, q1, q2, q3, K, V, kv_base, j0, offs_m, i_lo, T, Lv, qk_scale,
                                  RB, 1, D, BN, PREC)
    for j0 in range(b1, j_end, BN):
        m_i, l_i, acc = _fwd_tile(m_i, l_i, acc, q0, q1, q2, q3, K, V, kv_base, j0, offs_m, i_lo, T, Lv, qk_scale,
                                  RB, 0, D, BN, PREC)

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    offs_d = tl.arange(0, D)
    tl.store(O + q_base + offs_m[:, None] * D + offs_d[None, :], (acc / l_safe[:, None]).to(O.dtype.element_ty),
             mask=rmask[:, None])
    tl.store(LSE + bh.to(tl.int64) * T + offs_m, m_i + tl.log2(l_safe), mask=rmask)  # base-2 log-sum-exp


# --------------------------------------------------------------------------- backward: dq


@triton.jit
def _dq_tile(dq0, dq1, dq2, dq3, q0, q1, q2, q3, do, lse, Di, K, V, kv_base, j0, offs_m, rmask, i_lo, T, Lv,
             qk_scale, RB, C0: tl.constexpr, D: tl.constexpr, BN: tl.constexpr, PREC: tl.constexpr):
    offs_n = j0 + tl.arange(0, BN)
    kmask = offs_n < T
    BM: tl.constexpr = offs_m.shape[0]
    a = _count_le(Lv, _dmin(i_lo, j0, RB))
    s = _scores(q0, q1, q2, q3, K, kv_base, offs_n, kmask, a, C0, D, PREC)
    p = tl.exp2(s * qk_scale - lse[:, None])
    if (j0 + BN - 1 > i_lo) | (j0 + BN > T) | (i_lo + BM > T):
        p = tl.where((offs_m[:, None] >= offs_n[None, :]) & kmask[None, :] & rmask[:, None], p, 0.0)
    v = tl.load(V + kv_base + offs_n[:, None] * D + tl.arange(0, D)[None, :], mask=kmask[:, None], other=0.0)
    dp = tl.dot(do, tl.trans(v), input_precision=PREC)
    dsb = (p * (dp - Di[:, None])).to(do.dtype)
    if C0 <= 0:
        dq0 = _acc(dq0, dsb, _ld(K, kv_base, offs_n, kmask, 0, D), 0, C0, a, PREC)
    if C0 <= 1:
        dq1 = _acc(dq1, dsb, _ld(K, kv_base, offs_n, kmask, 1, D), 1, C0, a, PREC)
    if C0 <= 2:
        dq2 = _acc(dq2, dsb, _ld(K, kv_base, offs_n, kmask, 2, D), 2, C0, a, PREC)
    dq3 = _acc(dq3, dsb, _ld(K, kv_base, offs_n, kmask, 3, D), 3, C0, a, PREC)
    return dq0, dq1, dq2, dq3


@triton.autotune(configs=_configs("dq"), key=["T", "H", "G", "RB"], prune_configs_by={"early_config_prune": _prune})
@triton.jit
def _bwd_dq_kernel(Q, K, V, DO, LSE, DELTA, DQ, Lptr, sm_scale, qk_scale, T, H, G, RB,
                   D: tl.constexpr, P: tl.constexpr, PREC: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    hk = (bh % H) // G
    q_base = bh.to(tl.int64) * T * D
    kv_base = (b * (H // G) + hk).to(tl.int64) * T * D
    offs_m = pid_m * BM + tl.arange(0, BM)
    rmask = offs_m < T
    offs_d = tl.arange(0, D)
    q0 = _ld(Q, q_base, offs_m, rmask, 0, D)
    q1 = _ld(Q, q_base, offs_m, rmask, 1, D)
    q2 = _ld(Q, q_base, offs_m, rmask, 2, D)
    q3 = _ld(Q, q_base, offs_m, rmask, 3, D)
    do = tl.load(DO + q_base + offs_m[:, None] * D + offs_d[None, :], mask=rmask[:, None], other=0.0)
    lse = tl.load(LSE + bh.to(tl.int64) * T + offs_m, mask=rmask, other=0.0)
    Di = tl.load(DELTA + bh.to(tl.int64) * T + offs_m, mask=rmask, other=0.0)
    Lv = tl.load(Lptr + tl.arange(0, P))
    dq0 = tl.zeros([BM, 16], tl.float32)
    dq1 = tl.zeros([BM, 16], tl.float32)
    dq2 = tl.zeros([BM, 16], tl.float32)
    dq3 = tl.zeros([BM, 16], tl.float32)

    i_lo = pid_m * BM
    j_end = tl.minimum(i_lo + BM - 1, T - 1) + 1
    j_start, b3, b2, b1 = _key_segments(Lptr, i_lo, j_end, RB)
    for j0 in range(j_start, b3, BN):
        dq0, dq1, dq2, dq3 = _dq_tile(dq0, dq1, dq2, dq3, q0, q1, q2, q3, do, lse, Di, K, V, kv_base, j0, offs_m,
                                      rmask, i_lo, T, Lv, qk_scale, RB, 3, D, BN, PREC)
    for j0 in range(b3, b2, BN):
        dq0, dq1, dq2, dq3 = _dq_tile(dq0, dq1, dq2, dq3, q0, q1, q2, q3, do, lse, Di, K, V, kv_base, j0, offs_m,
                                      rmask, i_lo, T, Lv, qk_scale, RB, 2, D, BN, PREC)
    for j0 in range(b2, b1, BN):
        dq0, dq1, dq2, dq3 = _dq_tile(dq0, dq1, dq2, dq3, q0, q1, q2, q3, do, lse, Di, K, V, kv_base, j0, offs_m,
                                      rmask, i_lo, T, Lv, qk_scale, RB, 1, D, BN, PREC)
    for j0 in range(b1, j_end, BN):
        dq0, dq1, dq2, dq3 = _dq_tile(dq0, dq1, dq2, dq3, q0, q1, q2, q3, do, lse, Di, K, V, kv_base, j0, offs_m,
                                      rmask, i_lo, T, Lv, qk_scale, RB, 0, D, BN, PREC)
    cc = tl.arange(0, 16)[None, :]
    rows = DQ + q_base + offs_m[:, None] * D
    tl.store(rows + cc, (dq0 * sm_scale).to(DQ.dtype.element_ty), mask=rmask[:, None])
    tl.store(rows + 16 + cc, (dq1 * sm_scale).to(DQ.dtype.element_ty), mask=rmask[:, None])
    tl.store(rows + 32 + cc, (dq2 * sm_scale).to(DQ.dtype.element_ty), mask=rmask[:, None])
    tl.store(rows + 48 + cc, (dq3 * sm_scale).to(DQ.dtype.element_ty), mask=rmask[:, None])


# --------------------------------------------------------------------------- backward: dk, dv


@triton.jit
def _dkdv_tile(dk0, dk1, dk2, dk3, dv, k0, k1, k2, k3, v, Q, DO, LSE, DELTA, DQ, q_base, bh, i0, offs_n, kmask,
               j_lo, j_hi, T, Lv, qk_scale, RB, C0: tl.constexpr, D: tl.constexpr, BM: tl.constexpr,
               PREC: tl.constexpr, FUSED: tl.constexpr):
    offs_m = i0 + tl.arange(0, BM)
    rmask = offs_m < T
    do = tl.load(DO + q_base + offs_m[:, None] * D + tl.arange(0, D)[None, :], mask=rmask[:, None], other=0.0)
    lse = tl.load(LSE + bh.to(tl.int64) * T + offs_m, mask=rmask, other=0.0)
    Di = tl.load(DELTA + bh.to(tl.int64) * T + offs_m, mask=rmask, other=0.0)
    a = _count_le(Lv, _dmin(i0, j_lo, RB))
    # scores: query chunks loaded per tile, key chunks resident
    s = tl.zeros([BM, offs_n.shape[0]], tl.float32)
    if C0 <= 0:
        s = _qk(_ld(Q, q_base, offs_m, rmask, 0, D), k0, s, 0, C0, a, PREC)
    if C0 <= 1:
        s = _qk(_ld(Q, q_base, offs_m, rmask, 1, D), k1, s, 1, C0, a, PREC)
    if C0 <= 2:
        s = _qk(_ld(Q, q_base, offs_m, rmask, 2, D), k2, s, 2, C0, a, PREC)
    s = _qk(_ld(Q, q_base, offs_m, rmask, 3, D), k3, s, 3, C0, a, PREC)
    p = tl.exp2(s * qk_scale - lse[:, None])
    if (j_hi > i0) | (j_lo + offs_n.shape[0] > T) | (i0 + BM > T):
        p = tl.where((offs_m[:, None] >= offs_n[None, :]) & kmask[None, :] & rmask[:, None], p, 0.0)
    dv = tl.dot(tl.trans(p.to(do.dtype)), do, dv, input_precision=PREC)
    dp = tl.dot(do, tl.trans(v), input_precision=PREC)
    dst = tl.trans((p * (dp - Di[:, None])).to(do.dtype))
    if C0 <= 0:
        dk0 = _acc(dk0, dst, _ld(Q, q_base, offs_m, rmask, 0, D), 0, C0, a, PREC)
    if C0 <= 1:
        dk1 = _acc(dk1, dst, _ld(Q, q_base, offs_m, rmask, 1, D), 1, C0, a, PREC)
    if C0 <= 2:
        dk2 = _acc(dk2, dst, _ld(Q, q_base, offs_m, rmask, 2, D), 2, C0, a, PREC)
    dk3 = _acc(dk3, dst, _ld(Q, q_base, offs_m, rmask, 3, D), 3, C0, a, PREC)
    if FUSED:  # dQ_c += dS K_c, accumulated across key tiles with fp32 atomics
        dsb = tl.trans(dst)
        rows = DQ + q_base + offs_m[:, None] * D + tl.arange(0, 16)[None, :]
        zero = tl.zeros([BM, 16], tl.float32)
        if C0 <= 0:
            tl.atomic_add(rows, _acc(zero, dsb, k0, 0, C0, a, PREC), mask=rmask[:, None], sem="relaxed")
        if C0 <= 1:
            tl.atomic_add(rows + 16, _acc(zero, dsb, k1, 1, C0, a, PREC), mask=rmask[:, None], sem="relaxed")
        if C0 <= 2:
            tl.atomic_add(rows + 32, _acc(zero, dsb, k2, 2, C0, a, PREC), mask=rmask[:, None], sem="relaxed")
        tl.atomic_add(rows + 48, _acc(zero, dsb, k3, 3, C0, a, PREC), mask=rmask[:, None], sem="relaxed")
    return dk0, dk1, dk2, dk3, dv


@triton.autotune(configs=_configs("dkdv"), key=["T", "H", "G", "RB", "FUSED"],
                 prune_configs_by={"early_config_prune": _prune}, reset_to_zero=["DQ"])
@triton.jit
def _bwd_dkdv_kernel(Q, K, V, DO, LSE, DELTA, DK, DV, DQ, Lptr, sm_scale, qk_scale, T, H, G, RB,
                     D: tl.constexpr, P: tl.constexpr, PREC: tl.constexpr, FUSED: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr):
    """dK, dV per key tile; with FUSED also dQ (unscaled, fp32 atomics into DQ, chunked layout)."""
    pid_n = tl.program_id(0)
    bhk = tl.program_id(1)          # b * H_kv + kv head
    Hkv = H // G
    b = bhk // Hkv
    hk = bhk % Hkv
    kv_base = bhk.to(tl.int64) * T * D
    offs_n = pid_n * BN + tl.arange(0, BN)
    kmask = offs_n < T
    offs_d = tl.arange(0, D)
    k0 = _ld(K, kv_base, offs_n, kmask, 0, D)
    k1 = _ld(K, kv_base, offs_n, kmask, 1, D)
    k2 = _ld(K, kv_base, offs_n, kmask, 2, D)
    k3 = _ld(K, kv_base, offs_n, kmask, 3, D)
    v = tl.load(V + kv_base + offs_n[:, None] * D + offs_d[None, :], mask=kmask[:, None], other=0.0)
    Lv = tl.load(Lptr + tl.arange(0, P))
    dk0 = tl.zeros([BN, 16], tl.float32)
    dk1 = tl.zeros([BN, 16], tl.float32)
    dk2 = tl.zeros([BN, 16], tl.float32)
    dk3 = tl.zeros([BN, 16], tl.float32)
    dv = tl.zeros([BN, D], tl.float32)

    j_lo = pid_n * BN
    j_hi = j_lo + BN - 1
    i_start = (j_lo // BM) * BM
    s1, s2, s3, i_end = _query_segments(Lptr, j_lo, i_start, T, RB)
    for g in range(0, G):
        bh = b * H + hk * G + g
        q_base = bh.to(tl.int64) * T * D
        for i0 in range(i_start, s1, BM):
            dk0, dk1, dk2, dk3, dv = _dkdv_tile(dk0, dk1, dk2, dk3, dv, k0, k1, k2, k3, v, Q, DO, LSE, DELTA, DQ,
                                                q_base, bh, i0, offs_n, kmask, j_lo, j_hi, T, Lv, qk_scale, RB, 0, D,
                                                BM, PREC, FUSED)
        for i0 in range(s1, s2, BM):
            dk0, dk1, dk2, dk3, dv = _dkdv_tile(dk0, dk1, dk2, dk3, dv, k0, k1, k2, k3, v, Q, DO, LSE, DELTA, DQ,
                                                q_base, bh, i0, offs_n, kmask, j_lo, j_hi, T, Lv, qk_scale, RB, 1, D,
                                                BM, PREC, FUSED)
        for i0 in range(s2, s3, BM):
            dk0, dk1, dk2, dk3, dv = _dkdv_tile(dk0, dk1, dk2, dk3, dv, k0, k1, k2, k3, v, Q, DO, LSE, DELTA, DQ,
                                                q_base, bh, i0, offs_n, kmask, j_lo, j_hi, T, Lv, qk_scale, RB, 2, D,
                                                BM, PREC, FUSED)
        for i0 in range(s3, i_end, BM):
            dk0, dk1, dk2, dk3, dv = _dkdv_tile(dk0, dk1, dk2, dk3, dv, k0, k1, k2, k3, v, Q, DO, LSE, DELTA, DQ,
                                                q_base, bh, i0, offs_n, kmask, j_lo, j_hi, T, Lv, qk_scale, RB, 3, D,
                                                BM, PREC, FUSED)
    cc = tl.arange(0, 16)[None, :]
    rows = DK + kv_base + offs_n[:, None] * D
    tl.store(rows + cc, (dk0 * sm_scale).to(DK.dtype.element_ty), mask=kmask[:, None])
    tl.store(rows + 16 + cc, (dk1 * sm_scale).to(DK.dtype.element_ty), mask=kmask[:, None])
    tl.store(rows + 32 + cc, (dk2 * sm_scale).to(DK.dtype.element_ty), mask=kmask[:, None])
    tl.store(rows + 48 + cc, (dk3 * sm_scale).to(DK.dtype.element_ty), mask=kmask[:, None])
    tl.store(DV + kv_base + offs_n[:, None] * D + offs_d[None, :], dv.to(DV.dtype.element_ty), mask=kmask[:, None])


# --------------------------------------------------------------------------- launchers


def _check(q, k, v, L, rule_block):
    B, H, T, D = q.shape
    if not (k.shape == v.shape and k.shape[0] == B and k.shape[2] == T and k.shape[3] == D):
        raise ValueError(f"k, v must be [B, H_kv, T, d] matching q {tuple(q.shape)}: "
                         f"got {tuple(k.shape)}, {tuple(v.shape)}")
    if H % k.shape[1]:
        raise ValueError("H must be a multiple of H_kv")
    if D != HEAD_DIM:
        raise ValueError(f"these kernels are written for head_dim {HEAD_DIM} (4 chunks of 16 features)")
    if not (L.dtype == torch.int32 and L.numel() == D // 2):
        raise ValueError("L must be the int32 thresholds of the d / 2 pairs (hisa.rope.thresholds)")
    if not (rule_block >= 32 and rule_block % 32 == 0):
        raise ValueError("rule_block must be a multiple of 32 (kernel tiles divide it)")
    return B, H, T, D, H // k.shape[1]


@torch.library.custom_op("hisa::tile_attn_fwd", mutates_args=())
def tile_attn_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, L: torch.Tensor, rule_block: int,
                  fused_bwd: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> (o, lse2, q_chunked, k_chunked); the chunked copies are kept for the backward pass."""
    B, H, T, D, G = _check(q, k, v, L, rule_block)
    perm, _ = _perm(q.device)
    qc, kc, v = q.index_select(-1, perm).contiguous(), k.index_select(-1, perm).contiguous(), v.contiguous()
    o = _contiguous_like(q)
    lse = torch.empty(B, H, T, device=q.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(T, meta["BM"]), B * H)
    _fwd_kernel[grid](qc, kc, v, o, lse, L, D ** -0.5 * LOG2E, T, H, G, rule_block, D=D, P=D // 2, PREC=_prec(q))
    return o, lse, qc, kc


@tile_attn_fwd.register_fake
def _(q, k, v, L, rule_block, fused_bwd=True):
    B, H, T, D = q.shape
    return (_contiguous_like(q), torch.empty(B, H, T, dtype=torch.float32, device=q.device),
            _contiguous_like(q), _contiguous_like(k))


@torch.library.custom_op("hisa::tile_attn_bwd", mutates_args=())
def tile_attn_bwd(do: torch.Tensor, qc: torch.Tensor, kc: torch.Tensor, v: torch.Tensor, o: torch.Tensor,
                  lse: torch.Tensor, L: torch.Tensor, rule_block: int, fused: bool = True
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradients w.r.t. the original (unpermuted) q, k and v (fused: one backward kernel, dQ by atomics)."""
    v, o, do = v.contiguous(), o.contiguous(), do.contiguous()
    B, H, T, D, G = _check(qc, kc, v, L, rule_block)
    _, inv = _perm(qc.device)
    delta = (do.float() * o.float()).sum(-1)  # [B, H, T] = rowsum(dO * O)
    dqc, dkc, dv = _contiguous_like(qc), _contiguous_like(kc), _contiguous_like(v)
    scales = (D ** -0.5, D ** -0.5 * LOG2E)
    consts = dict(D=D, P=D // 2, PREC=_prec(qc))
    dkdv_grid = lambda meta: (triton.cdiv(T, meta["BN"]), B * (H // G))
    if fused:
        dq_acc = torch.zeros(qc.shape, device=qc.device, dtype=torch.float32)
        _bwd_dkdv_kernel[dkdv_grid](qc, kc, v, do, lse, delta, dkc, dv, dq_acc, L, *scales, T, H, G, rule_block,
                                    FUSED=True, **consts)
        dqc = (dq_acc * scales[0]).to(qc.dtype)
    else:
        _bwd_dq_kernel[lambda meta: (triton.cdiv(T, meta["BM"]), B * H)](
            qc, kc, v, do, lse, delta, dqc, L, *scales, T, H, G, rule_block, **consts)
        _bwd_dkdv_kernel[dkdv_grid](qc, kc, v, do, lse, delta, dkc, dv, dkc, L, *scales, T, H, G, rule_block,
                                    FUSED=False, **consts)  # DQ unused
    return dqc.index_select(-1, inv), dkc.index_select(-1, inv), dv


@tile_attn_bwd.register_fake
def _(do, qc, kc, v, o, lse, L, rule_block, fused=True):
    return _contiguous_like(qc), _contiguous_like(kc), _contiguous_like(v)


def _setup_context(ctx, inputs, output):
    q, k, v, L, rule_block, fused_bwd = inputs
    o, lse, qc, kc = output
    ctx.save_for_backward(qc, kc, v, o, lse, L)
    ctx.rule_block, ctx.fused_bwd = rule_block, fused_bwd


def _backward(ctx, do, dlse, dqc_unused, dkc_unused):
    qc, kc, v, o, lse, L = ctx.saved_tensors
    dq, dk, dv = tile_attn_bwd(do, qc, kc, v, o, lse, L, ctx.rule_block, ctx.fused_bwd)
    return dq, dk, dv, None, None, None


tile_attn_fwd.register_autograd(_backward, setup_context=_setup_context)


def hisa_triton_block(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, L: torch.Tensor,
                               rule_block: int = 128, fused_bwd: bool = True) -> torch.Tensor:
    """q [B, H, T, 64], k, v [B, H_kv, T, 64] (rotated) -> [B, H, T, 64] under the block rule;
    L = hisa.rope.thresholds(theta, max_angle_deg). Autocast handling as in triton_exact."""
    dev = q.device.type
    if torch.is_autocast_enabled(dev):
        dt = torch.get_autocast_dtype(dev)
        q, k, v = q.to(dt), k.to(dt), v.to(dt)
    elif not (q.dtype == k.dtype == v.dtype):
        raise TypeError(f"q/k/v dtypes differ: {q.dtype}, {k.dtype}, {v.dtype}")
    o, _, _, _ = tile_attn_fwd(q, k, v, L, rule_block, fused_bwd)
    return o
