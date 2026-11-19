"""Triton kernels (exact rule and block rule) against the PyTorch reference: outputs and gradients.

Runs on an NVIDIA GPU, or on CPU with the Triton interpreter:
    TRITON_INTERPRET=1 pytest tests/test_triton.py
"""

import math
import os

import pytest
import torch

from hisa import HiSAAttention, hisa_attention_reference, rope, sdpa_attention

DEVICE = "cuda" if torch.cuda.is_available() else ("cpu" if os.environ.get("TRITON_INTERPRET") == "1" else None)
pytestmark = pytest.mark.skipif(DEVICE is None, reason="needs a GPU or TRITON_INTERPRET=1")

if DEVICE is not None:
    from hisa.kernels.triton_block import hisa_triton_block
    from hisa.kernels.triton_exact import hisa_triton_exact


def rand_qkv(B, H, Hkv, T, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    mk = lambda h: torch.randn(B, h, T, 64, generator=g, dtype=torch.float32).to(DEVICE, dtype).requires_grad_(True)
    return mk(H), mk(Hkv), mk(Hkv)


def run(fn, q, k, v, theta, angle, rule_block=None):
    out = fn(q, k, v)
    ref = hisa_attention_reference(q.double(), k.double(), v.double(), theta.to(DEVICE), angle, rule_block, 16)
    g = torch.randn_like(ref)
    return out, ref, torch.autograd.grad(out, (q, k, v), g.to(out.dtype)), torch.autograd.grad(ref, (q, k, v), g)


def check(out, ref, grads, grads_ref):
    torch.testing.assert_close(out.double(), ref, rtol=1e-4, atol=1e-5)
    for name, a, b in zip("qkv", grads, grads_ref):
        torch.testing.assert_close(a.double(), b.double(), rtol=1e-4, atol=1e-4, msg=f"grad {name}")


# base 30: most pairs are cut within a few tokens (tiles without any kept pair); base 1000: every chunk level
# occurs at T <= 300; base 10^4: the SmolLM / Llama default
@pytest.mark.parametrize("base", [30.0, 1000.0, 10000.0])
@pytest.mark.parametrize("angle", [45.0, 90.0, 720.0, math.inf])
@pytest.mark.parametrize("T", [37, 130])
def test_exact_fp32_matches_reference(base, angle, T):
    theta = rope.inv_freq(64, base)
    L = rope.thresholds(theta, angle, DEVICE)
    q, k, v = rand_qkv(2, 4, 2, T, torch.float32)
    check(*run(lambda q, k, v: hisa_triton_exact(q, k, v, L), q, k, v, theta, angle))


@pytest.mark.parametrize("base", [30.0, 1000.0, 10000.0])
@pytest.mark.parametrize("angle", [90.0, 360.0, math.inf])
@pytest.mark.parametrize("T,rule_block", [(130, 32), (300, 64)])
@pytest.mark.parametrize("fused", [True, False])
def test_block_fp32_matches_reference(base, angle, T, rule_block, fused):
    theta = rope.inv_freq(64, base)
    L = rope.thresholds(theta, angle, DEVICE)
    q, k, v = rand_qkv(2, 4, 2, T, torch.float32, seed=1)
    check(*run(lambda q, k, v: hisa_triton_block(q, k, v, L, rule_block, fused), q, k, v, theta, angle, rule_block))


def test_realistic_shape_fp32():
    """SmolLM layout (d = 64, base 10^4, GQA 9 / 3) at 90 degrees."""
    T = 600 if DEVICE == "cuda" else 300
    theta = rope.inv_freq(64)
    L = rope.thresholds(theta, 90.0, DEVICE)
    q, k, v = rand_qkv(1, 9, 3, T, torch.float32, seed=2)
    check(*run(lambda q, k, v: hisa_triton_exact(q, k, v, L), q, k, v, theta, 90.0))


@pytest.mark.skipif(DEVICE != "cuda", reason="bf16 tensor-core path: GPU only")
@pytest.mark.parametrize("impl", ["exact", "block"])
def test_bf16_error_is_at_bf16_level(impl):
    theta = rope.inv_freq(64)
    L = rope.thresholds(theta, 90.0, DEVICE)
    rb = None if impl == "exact" else 128
    fn = (lambda q, k, v: hisa_triton_exact(q, k, v, L)) if impl == "exact" else \
        (lambda q, k, v: hisa_triton_block(q, k, v, L, 128))
    q, k, v = rand_qkv(2, 9, 3, 2048, torch.bfloat16, seed=3)
    out, ref, grads, grads_ref = run(fn, q, k, v, theta, 90.0, rb)
    rel = lambda x, r: ((x.double() - r.double()).norm() / r.double().norm()).item()
    assert rel(out, ref) < 1e-2
    for a, b in zip(grads, grads_ref):
        assert rel(a, b) < 2e-2


def test_inf_angle_is_dense_attention():
    q, k, v = (x.detach() for x in rand_qkv(1, 9, 3, 300, torch.float32, seed=4))
    L = rope.thresholds(rope.inv_freq(64), math.inf, DEVICE)
    dense = sdpa_attention(q, k, v)
    torch.testing.assert_close(hisa_triton_exact(q, k, v, L), dense, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(hisa_triton_block(q, k, v, L, 64), dense, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("impl", ["exact", "block"])
@pytest.mark.parametrize("c", [1, 63, 64, 65, 200])
def test_no_causal_leak(impl, c):
    """Changing positions >= c (queries, keys and values) leaves the outputs of positions < c bit-identical."""
    theta = rope.inv_freq(64, 30.0)
    L = rope.thresholds(theta, 90.0, DEVICE)
    fn = (lambda q, k, v: hisa_triton_exact(q, k, v, L)) if impl == "exact" else \
        (lambda q, k, v: hisa_triton_block(q, k, v, L, 64))
    q, k, v = (x.detach() for x in rand_qkv(1, 9, 3, 300, torch.float32, seed=5))
    out0 = fn(q, k, v)
    gen = torch.Generator().manual_seed(c)
    q1, k1, v1 = (x.clone() for x in (q, k, v))
    for x in (q1, k1, v1):
        x[..., c:, :] = 30.0 * torch.randn(x[..., c:, :].shape, generator=gen).to(DEVICE)
    assert torch.equal(fn(q1, k1, v1)[..., :c, :], out0[..., :c, :])


def test_module_dispatch():
    theta = rope.inv_freq(64, 1000.0)
    q, k, v = (x.detach() for x in rand_qkv(1, 9, 3, 200, torch.float32, seed=6))
    L = rope.thresholds(theta, 90.0, DEVICE)
    torch.testing.assert_close(HiSAAttention(theta, impl="triton")(q, k, v), hisa_triton_exact(q, k, v, L),
                               rtol=0, atol=0)
    torch.testing.assert_close(HiSAAttention(theta, impl="triton_block", rule_block=64)(q, k, v),
                               hisa_triton_block(q, k, v, L, 64), rtol=0, atol=0)


def test_fake_kernels_match_real_layout():
    """torch.compile traces the custom ops through their fake kernels: same shapes, dtypes and strides."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    from hisa.kernels.triton_exact import triton_attn_fwd

    q, k, v = (x.detach() for x in rand_qkv(1, 4, 2, 64, torch.float32, seed=7))
    q = q.transpose(1, 2).contiguous().transpose(1, 2)  # non-contiguous input, as after RoPE in a model
    L = rope.thresholds(rope.inv_freq(64), 90.0, DEVICE)
    real = triton_attn_fwd(q, k, v, L)
    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        fake = triton_attn_fwd(*(mode.from_tensor(x) for x in (q, k, v, L)))
    for a, b in zip(real, fake):
        assert a.shape == b.shape and a.dtype == b.dtype and a.stride() == b.stride()


def test_input_checks_raise_value_error():
    """The wrappers validate shapes with ValueError (not assert: kept under python -O), before any kernel launch."""
    pytest.importorskip("triton")
    from hisa.kernels.triton_block import hisa_triton_block
    from hisa.kernels.triton_exact import hisa_triton_exact

    L = rope.thresholds(rope.inv_freq(64), 90.0)
    q, k = torch.randn(1, 2, 8, 64), torch.randn(1, 1, 8, 64)
    bad = [(torch.randn(1, 2, 8, 32), torch.randn(1, 1, 8, 32), L),  # head dim 32
           (q, torch.randn(1, 1, 9, 64), L),  # T differs
           (torch.randn(1, 3, 8, 64), torch.randn(1, 2, 8, 64), L),  # H not a multiple of H_kv
           (q, k, L.long())]  # thresholds not int32
    for q_, k_, L_ in bad:
        with pytest.raises(ValueError):
            hisa_triton_exact(q_, k_, k_, L_)
        with pytest.raises(ValueError):
            hisa_triton_block(q_, k_, k_, L_, 128)
    with pytest.raises(ValueError):
        hisa_triton_block(q, k, k, L, 48)
