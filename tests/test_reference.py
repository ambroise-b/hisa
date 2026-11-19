"""The rule (windows, thresholds, masks) and the PyTorch reference. CPU, fp64."""

import math

import pytest
import torch

from hisa import HiSAAttention, hisa_attention, hisa_attention_reference, rope, sdpa_attention


def rand_qkv(B, H, Hkv, T, d=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, h, T, d, generator=g, dtype=torch.float64) for h in (H, Hkv, Hkv)]


def naive(q, k, v, theta, angle, rule_block=None):
    """Triple loop over (i, j, p) on the HF pair layout: the definition, written out."""
    B, H, T, d = q.shape
    G, P = H // k.shape[1], d // 2
    w = rope.window(theta, angle)
    out = torch.zeros_like(q)
    for b in range(B):
        for h in range(H):
            qh, kh, vh = q[b, h], k[b, h // G], v[b, h // G]
            for i in range(T):
                s = torch.full((T,), -math.inf, dtype=torch.float64)
                for j in range(i + 1):
                    dist = i - j
                    if rule_block is not None:
                        dist = max((i // rule_block - j // rule_block) * rule_block - (rule_block - 1), 0)
                    kept = [p for p in range(P) if dist < w[p]]
                    if kept:
                        s[j] = sum(qh[i, p] * kh[j, p] + qh[i, p + P] * kh[j, p + P] for p in kept) / math.sqrt(d)
                out[b, h, i] = torch.softmax(s, 0) @ vh
    return out


def test_window_and_reach():
    theta = rope.inv_freq(64, 1e4)
    w = rope.window(theta, 90.0)
    assert w[0] == pytest.approx(math.pi / 2)
    assert rope.reach(theta, 90.0) == pytest.approx(11_800, rel=1e-2)        # paper: 11.8K at b = 10^4
    assert rope.reach(rope.inv_freq(64, 2e4), 90.0) == pytest.approx(23_100, rel=1e-2)
    assert torch.isinf(rope.window(theta, math.inf)).all()


def test_thresholds_match_window():
    theta = rope.inv_freq(64, 1e4)
    for angle in [45.0, 90.0, 360.0]:
        L, w = rope.thresholds(theta, angle), rope.window(theta, angle)
        for D in range(0, 30000, 7):
            assert torch.equal(torch.tensor(D) < L, torch.tensor(float(D), dtype=torch.float64) < w)
    assert (rope.thresholds(theta, math.inf) == rope.BIG).all()


def test_chunk_permutation():
    perm = rope.chunk_permutation(64)
    assert sorted(perm.tolist()) == list(range(64))
    for c in range(4):  # chunk c holds pairs 8c .. 8c+7, both features of each
        assert sorted(f % 32 for f in perm[16 * c: 16 * c + 16].tolist()) == sorted(list(range(8 * c, 8 * c + 8)) * 2)


@pytest.mark.parametrize("rule_block", [None, 4])
@pytest.mark.parametrize("base,angle", [(30.0, 90.0), (1000.0, 45.0), (10000.0, 90.0)])
def test_reference_matches_naive(base, angle, rule_block):
    theta = rope.inv_freq(16, base)
    q, k, v = rand_qkv(1, 2, 1, 13, d=16, seed=1)
    ref = hisa_attention_reference(q, k, v, theta, angle, rule_block, block_size=4)
    torch.testing.assert_close(ref, naive(q, k, v, theta, angle, rule_block), rtol=1e-12, atol=1e-12)


def test_block_rule_keeps_every_exact_pair():
    theta = rope.inv_freq(64, 1000.0)
    exact = rope.pair_masks(theta, 90.0, 300)
    block = rope.pair_masks(theta, 90.0, 300, rule_block=32)
    assert (block | ~exact).all() and block.sum() > exact.sum()


@pytest.mark.parametrize("rule_block", [None, 16])
def test_inf_angle_is_dense_attention(rule_block):
    q, k, v = rand_qkv(1, 4, 2, 50, seed=3)
    out = hisa_attention_reference(q, k, v, rope.inv_freq(64), math.inf, rule_block)
    torch.testing.assert_close(out, sdpa_attention(q, k, v), rtol=1e-10, atol=1e-10)


def test_sparsity_changes_the_output():
    q, k, v = rand_qkv(1, 2, 1, 80, seed=4)
    theta = rope.inv_freq(64, 30.0)
    assert (hisa_attention_reference(q, k, v, theta, 90.0) - sdpa_attention(q, k, v)).abs().max() > 1e-3


def test_keys_beyond_the_reach_are_ignored():
    theta = rope.inv_freq(64, 30.0)
    R = math.ceil(rope.reach(theta, 90.0))
    T = R + 20
    q, k, v = rand_qkv(1, 2, 1, T, seed=5)
    out = hisa_attention_reference(q, k, v, theta, 90.0)
    v2 = v.clone()
    v2[..., 0, :] += 100.0  # key 0 is beyond the reach of the last queries
    out2 = hisa_attention_reference(q, k, v2, theta, 90.0)
    torch.testing.assert_close(out2[..., R:, :], out[..., R:, :], rtol=0, atol=0)


@pytest.mark.parametrize("c", [1, 17, 40])
def test_no_causal_leak(c):
    theta = rope.inv_freq(64, 30.0)
    q, k, v = rand_qkv(1, 4, 2, 60, seed=6)
    out0 = hisa_attention_reference(q, k, v, theta, 90.0)
    q1, k1, v1 = (x.clone() for x in (q, k, v))
    for x in (q1, k1, v1):
        x[..., c:, :] = 30.0 * torch.randn(x[..., c:, :].shape, dtype=torch.float64)
    assert torch.equal(hisa_attention_reference(q1, k1, v1, theta, 90.0)[..., :c, :], out0[..., :c, :])


def test_gradcheck():
    theta = rope.inv_freq(16, 30.0)
    q, k, v = (x.requires_grad_(True) for x in rand_qkv(1, 2, 1, 11, d=16, seed=7))
    assert torch.autograd.gradcheck(lambda q, k, v: hisa_attention_reference(q, k, v, theta, 90.0, block_size=4),
                                    (q, k, v))


def test_api_dispatch_and_checks():
    q, k, v = rand_qkv(1, 2, 1, 20, seed=8)
    theta = rope.inv_freq(64, 30.0)
    ref = hisa_attention_reference(q, k, v, theta, 90.0)
    torch.testing.assert_close(hisa_attention(q, k, v, theta, impl="reference"), ref, rtol=0, atol=0)
    torch.testing.assert_close(HiSAAttention(theta, impl="reference")(q, k, v), ref, rtol=0, atol=0)
    with pytest.raises(ValueError):
        hisa_attention(q, k, v, theta, impl="triton", rule_block=128)
    with pytest.raises(ValueError):
        hisa_attention(q, k, v, theta, impl="triton_block")
    with pytest.raises(ValueError):
        hisa_attention(q, k, v, theta, impl="flash", rule_block=64)
    with pytest.raises(ValueError):
        hisa_attention(q, k, v, theta, impl="nope")
    for rb in (0, 48, 100):  # triton_block: the kernel tiles must divide rule_block (checked before any kernel)
        with pytest.raises(ValueError):
            hisa_attention(q, k, v, theta, impl="triton_block", rule_block=rb)
