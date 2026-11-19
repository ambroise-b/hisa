"""Hugging Face integration: patch_llama keeps the weights and, at an infinite budget, the model's function."""

import math
import os

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from hisa.integrations.hf import patch_llama  # noqa: E402

GPU = torch.cuda.is_available()


def tiny(seed=0, **kw):
    cfg = LlamaConfig(vocab_size=97, hidden_size=256, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128, use_cache=False, **kw)
    torch.manual_seed(seed)
    return LlamaForCausalLM(cfg)


def test_keeps_state_dict_keys():
    model = tiny()
    keys = set(model.state_dict())
    patch_llama(model, 90.0, impl="reference")
    assert set(model.state_dict()) == keys


def test_inf_budget_is_the_original_model():
    ref, model = tiny(), tiny()
    patch_llama(model, math.inf, impl="reference")
    ids = torch.randint(0, 97, (2, 40))
    with torch.no_grad():
        torch.testing.assert_close(model(input_ids=ids).logits, ref(input_ids=ids).logits, rtol=1e-4, atol=1e-4)


def test_finite_budget_changes_the_model():
    ref, model = tiny(), tiny()
    patch_llama(model, 90.0, impl="reference", rule_theta=1.0 / (30.0 ** (torch.arange(0, 64, 2) / 64)))
    ids = torch.randint(0, 97, (2, 60))
    with torch.no_grad():
        assert (model(input_ids=ids).logits - ref(input_ids=ids).logits).abs().max() > 1e-3


@pytest.mark.skipif(not GPU and os.environ.get("TRITON_INTERPRET") != "1", reason="Triton: GPU or interpreter")
@pytest.mark.parametrize("impl,rule_block", [("triton", None), ("triton_block", 64)])
def test_triton_model_matches_reference_model(impl, rule_block):
    dev = "cuda" if GPU else "cpu"
    a, b = tiny(head_dim=64), tiny(head_dim=64)
    patch_llama(a, 90.0, impl="reference", rule_block=rule_block)
    patch_llama(b, 90.0, impl=impl, rule_block=rule_block)
    a, b = a.to(dev), b.to(dev)
    ids = torch.randint(0, 97, (2, 128), device=dev)
    with torch.no_grad():
        torch.testing.assert_close(b(input_ids=ids).logits, a(input_ids=ids).logits, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not GPU, reason="torch.compile + Triton: GPU only")
def test_compiled_model_with_triton_attention():
    model = tiny(head_dim=64).cuda()
    patch_llama(model, 90.0, impl="triton")
    ids = torch.randint(0, 97, (2, 64), device="cuda")
    w = model.model.layers[0].self_attn.q_proj.weight
    eager = model(input_ids=ids).logits
    ge = torch.autograd.grad(eager.float().square().mean(), w)[0]
    compiled = torch.compile(model)(input_ids=ids).logits
    gc = torch.autograd.grad(compiled.float().square().mean(), w)[0]
    torch.testing.assert_close(compiled, eager, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(gc, ge, rtol=1e-3, atol=1e-4)


def test_padding_mask_raises():
    model = tiny()
    patch_llama(model, 90.0, impl="reference")
    ids = torch.randint(0, 97, (2, 24))
    full = torch.ones(2, 24, dtype=torch.long)
    with torch.no_grad():
        model(input_ids=ids, attention_mask=full)  # no padding: fine (transformers passes no mask or a causal one)
        padded = full.clone()
        padded[0, :5] = 0  # left padding
        with pytest.raises(ValueError, match="padding"):
            model(input_ids=ids, attention_mask=padded)


def test_check_no_padding_mask_forms():
    from hisa.integrations.hf import check_no_padding

    T = 6
    causal = torch.ones(T, T, dtype=torch.bool).tril()[None, None]
    check_no_padding(None)
    check_no_padding(torch.ones(2, T))
    check_no_padding(causal)  # 4D bool, True = keep
    check_no_padding(torch.zeros(1, 1, T, T).masked_fill(~causal, -1e9))  # 4D additive, 0 = keep
    pad = causal.clone()
    pad[..., 0] = False  # key 0 masked for every query
    for m in (torch.tensor([[0, 1, 1, 1, 1, 1]]), pad, torch.zeros(1, 1, T, T).masked_fill(~pad, -1e9)):
        with pytest.raises(ValueError):
            check_no_padding(m)


def test_decode_rule_from_model():
    pytest.importorskip("triton")
    from types import SimpleNamespace

    from hisa.integrations.hf_decode import decode_rule
    from hisa.rope import inv_freq

    theta = inv_freq(64, 1e4)
    flash = SimpleNamespace(hisa={"max_angle_deg": 90.0, "impl": "flash", "rule_block": 128, "chunked": True,
                                  "rule_theta": theta})
    a, th, c = decode_rule(flash, None, None, None)
    assert (a, c) == (90.0, True) and torch.equal(th, theta)
    assert decode_rule(flash, 90.0, theta, True)[0] == 90.0  # agreeing arguments are fine
    for kw in ({"max_angle_deg": 45.0}, {"chunked": False}, {"rule_theta": inv_freq(64, 500.0)}):
        args = {"max_angle_deg": None, "rule_theta": None, "chunked": None, **kw}
        with pytest.raises(ValueError):
            decode_rule(flash, **args)
    exact = SimpleNamespace(hisa={**flash.hisa, "impl": "triton", "rule_block": None, "chunked": False})
    with pytest.raises(ValueError, match="block rule"):
        decode_rule(exact, None, None, None)
    dense = SimpleNamespace(hisa={**exact.hisa, "max_angle_deg": math.inf})
    assert decode_rule(dense, None, None, None)[0] == math.inf  # dense: any impl
    plain = SimpleNamespace()  # not patched: arguments or defaults
    assert decode_rule(plain, None, None, None) == (90.0, None, False)
    assert decode_rule(plain, 60.0, None, True) == (60.0, None, True)
