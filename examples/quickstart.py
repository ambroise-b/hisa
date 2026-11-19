"""HiSA in 20 lines: compare every available implementation with the reference on random inputs.

    python examples/quickstart.py
"""

import math

import torch

from hisa import hisa_attention, hisa_attention_reference, rope

dev = "cuda" if torch.cuda.is_available() else "cpu"
B, H, H_kv, T, d = 1, 9, 3, 1024, 64
g = torch.Generator().manual_seed(0)
q, k, v = (torch.randn(B, h, T, d, generator=g).to(dev) for h in (H, H_kv, H_kv))
theta = rope.inv_freq(d, base=10_000.0)

print(f"windows (tokens) of pairs 0, 8, 16, 24, 31: {[round(x) for x in rope.window(theta, 90.0)[::8].tolist()]}")
print(f"reach: {rope.reach(theta, 90.0):.0f} tokens")

ref = hisa_attention_reference(q.double(), k.double(), v.double(), theta, 90.0)
dense = hisa_attention_reference(q.double(), k.double(), v.double(), theta, math.inf)
print(f"HiSA vs dense attention, max difference: {(ref - dense).abs().max():.3f}")
if dev == "cuda":
    out = hisa_attention(q, k, v, theta, 90.0, impl="triton")
    print(f"Triton (exact rule), max error vs reference: {(out.double() - ref).abs().max():.1e}")
    try:
        out = hisa_attention(q.bfloat16(), k.bfloat16(), v.bfloat16(), theta, 90.0, impl="flash")
        ref_b = hisa_attention_reference(q.double(), k.double(), v.double(), theta, 90.0, rule_block=128)
        print(f"Flash-HiSA (block rule, bf16), max error vs reference: {(out.double() - ref_b).abs().max():.1e}")
    except ImportError as e:
        print(f"Flash-HiSA not available: {e}")
