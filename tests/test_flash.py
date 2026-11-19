"""Flash-HiSA (FlashAttention-3 fork, Hopper): training kernels and decoding against the reference.

Needs an sm90 GPU and the fork built (scripts/build_flash.sh); skipped otherwise. bf16: the error of Flash-HiSA is
compared with the error of the Triton block-rule kernel on the same inputs (both against the fp64 reference).
"""

import math

import pytest
import torch

from hisa import hisa_attention_reference, rope

HAVE_FLASH = False
if torch.cuda.is_available():
    try:
        from hisa.kernels.flash import interface

        interface()
        HAVE_FLASH = True
    except Exception:  # noqa: BLE001  (not built, or not an sm90 GPU)
        pass
pytestmark = pytest.mark.skipif(not HAVE_FLASH, reason="needs an sm90 GPU and the Flash-HiSA build")

if HAVE_FLASH:
    from hisa.kernels.flash import flash_dense_attention, hisa_flash, sparse_table, table_with_header
    from hisa.kernels.flash_decode import HiSAKVCache
    from hisa.kernels.triton_block import hisa_triton_block

DEV = "cuda"
rel = lambda x, r: ((x.double() - r.double()).norm() / r.double().norm()).item()


def qkv(B, H, Hkv, T, seed=0, grad=False):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, h, T, 64, generator=g).to(DEV, torch.bfloat16).requires_grad_(grad) for h in (H, Hkv, Hkv)]


def test_sparse_table():
    L = rope.thresholds(rope.inv_freq(64), 90.0).tolist()
    t = sparse_table(L, 128, 16384)
    assert t[0] == 0 and t == sorted(t) and all(a < 32 for a in t)
    h = table_with_header(t, L)
    assert h[36:] == t and h[3] == len(t) and h[4:36] == L
    assert h[:3] == [2, 2, 11]  # the paper's nd_1..3 (b = 10^4, 90 degrees)
    assert len(t) == (L[-1] + 2 * 128 - 2) // 128  # blocks closer than the first one with dmin >= L_31


# base 30: blocks with no kept pair; base 1000 / 2600 / 10^4: all chunk levels; T not a multiple of the tiles
@pytest.mark.parametrize("base", [30.0, 1000.0, 2600.0, 10000.0])
@pytest.mark.parametrize("T", [300, 1000, 4096])
@pytest.mark.parametrize("angle", [90.0, 360.0])
def test_forward_backward_match_reference(base, T, angle):
    q, k, v = qkv(2, 9, 3, T, seed=7, grad=True)
    theta = rope.inv_freq(64, base)
    L = rope.thresholds(theta, angle)
    ref = hisa_attention_reference(q.double(), k.double(), v.double(), theta.to(DEV), angle, 128, 128)
    g = torch.randn(ref.shape, generator=torch.Generator().manual_seed(1)).to(DEV)
    gr = torch.autograd.grad(ref, (q, k, v), g.double())
    errs = {}
    for name, fn in [("flash", lambda: hisa_flash(q, k, v, L.tolist(), 128)),
                     ("triton", lambda: hisa_triton_block(q, k, v, L.to(DEV), 128))]:
        out = fn()
        grads = torch.autograd.grad(out, (q, k, v), g.to(out.dtype))
        errs[name] = [rel(out, ref)] + [rel(a, b) for a, b in zip(grads, gr)]
    for what, ef, et in zip(["out", "dq", "dk", "dv"], errs["flash"], errs["triton"]):
        assert ef <= 1.5 * et + 2e-3, (what, ef, et)


def test_inf_angle_equals_dense_flash():
    """Nothing dropped: Flash-HiSA equals dense FlashAttention-3 of the same build (on permuted features)."""
    q, k, v = qkv(2, 9, 3, 2048, seed=1)
    L = rope.thresholds(rope.inv_freq(64), math.inf).tolist()
    perm = rope.chunk_permutation().to(DEV)
    out = hisa_flash(q[..., perm], k[..., perm], v, L, chunked=True)
    torch.testing.assert_close(out, flash_dense_attention(q[..., perm], k[..., perm], v), rtol=1e-2, atol=1e-2)


def test_chunked_layout_equals_hf_layout():
    q, k, v = qkv(1, 9, 3, 1000, seed=3, grad=True)
    L = rope.thresholds(rope.inv_freq(64), 90.0).tolist()
    perm = rope.chunk_permutation().to(DEV)
    g = torch.randn(q.shape, generator=torch.Generator().manual_seed(9)).to(DEV, q.dtype)
    oh = hisa_flash(q, k, v, L)
    gh = torch.autograd.grad(oh, (q, k, v), g)
    qc, kc = (x.detach()[..., perm].requires_grad_(True) for x in (q, k))
    vc = v.detach().requires_grad_(True)
    oc = hisa_flash(qc, kc, vc, L, chunked=True)
    gc = torch.autograd.grad(oc, (qc, kc, vc), g)
    torch.testing.assert_close(oc, oh, rtol=0, atol=0)
    torch.testing.assert_close(gc[0], gh[0][..., perm], rtol=1e-2, atol=1e-2)  # dQ: fp32 atomics, order varies
    torch.testing.assert_close(gc[1], gh[1][..., perm], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(gc[2], gh[2], rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("c", [1, 127, 128, 129, 700])
def test_no_causal_leak(c):
    theta = rope.inv_freq(64, 1000.0)
    L = rope.thresholds(theta, 90.0).tolist()
    q, k, v = qkv(1, 9, 3, 1000, seed=5)
    out0 = hisa_flash(q, k, v, L)
    q1, k1, v1 = (x.clone() for x in (q, k, v))
    gen = torch.Generator().manual_seed(c)
    for x in (q1, k1, v1):
        x[..., c:, :] = (30.0 * torch.randn(x[..., c:, :].shape, generator=gen)).to(DEV, x.dtype)
    assert torch.equal(hisa_flash(q1, k1, v1, L)[..., :c, :], out0[..., :c, :])


# ----------------------------------------------------------------------------------------------- decoding


def test_ring_lengths_of_the_paper():
    c = HiSAKVCache(1, 1, 9, 3, 4096, rope.inv_freq(64), 90.0, DEV)
    assert c.R == [256, 256, 1408, 4096] and c.RV == 4096
    c = HiSAKVCache(1, 1, 9, 3, 16384, rope.inv_freq(64, 2e4), 90.0, DEV)
    assert c.R == [256, 384, 2176, 16384]


@pytest.mark.parametrize("num_splits", [0, 1, 3])
@pytest.mark.parametrize("base", [10000.0, 1000.0, 30.0])
@pytest.mark.parametrize("angle", [45.0, 90.0, math.inf])
def test_decode_matches_reference(angle, base, num_splits):
    B, H, Hkv, T = 2, 9, 3, 640
    q, k, v = qkv(B, H, Hkv, T)
    theta = rope.inv_freq(64, base)
    cache = HiSAKVCache(1, B, H, Hkv, T, theta, angle, DEV, num_splits=num_splits)
    t = torch.zeros(1, device=DEV, dtype=torch.int32)
    out = []
    for i in range(T):
        t.fill_(i)
        cache.append(0, cache.to_chunked(k[:, :, i]), v[:, :, i], t)
        out.append(cache.attend(0, cache.to_chunked(q[:, :, i]).contiguous(), t))
    out = torch.stack(out, 2)
    ref = hisa_attention_reference(q.double(), k.double(), v.double(), theta.to(DEV), angle, 128)
    assert torch.isfinite(out).all() and rel(out, ref) < 1e-2, rel(out, ref)


def test_decoder_matches_full_forward():
    """A tiny Llama trained-from-scratch layout (chunked, RoPE base 100: rings shorter than T): teacher-forced
    decoding with the ring-buffer cache gives the loss of the full forward pass."""
    import torch.nn.functional as F
    from transformers import LlamaConfig, LlamaForCausalLM

    from hisa.integrations.hf import patch_llama
    from hisa.integrations.hf_decode import HiSADecoder

    cfg = LlamaConfig(vocab_size=97, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024, use_cache=False,
                      rope_theta=100.0)
    cfg.rope_parameters = {"rope_theta": 100.0, "rope_type": "default"}
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    patch_llama(model, 90.0, impl="flash", chunked=True)
    model = model.to(DEV, torch.bfloat16).eval()
    T, B = 384, 3
    x = torch.randint(0, 97, (B, T + 1), generator=torch.Generator().manual_seed(1)).to(DEV)
    with torch.no_grad():
        logits = model(input_ids=x[:, :-1]).logits.float()
    ref = F.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten(), reduction="none").view(B, T)
    dec, _ = HiSADecoder(model, B, T, 90.0, chunked=True).teacher_forced(x)
    assert (dec - ref).abs().mean().item() < 2e-2, (dec - ref).abs().mean().item()
