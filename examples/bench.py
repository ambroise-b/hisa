"""Forward + backward time of HiSA against dense attention, per call (bf16, causal).

    python examples/bench.py --T 4096 16384
"""

import argparse

import torch
import torch.nn.functional as F

from hisa import HiSAAttention, rope

p = argparse.ArgumentParser()
p.add_argument("--T", type=int, nargs="+", default=[2048, 4096, 16384])
p.add_argument("--tokens", type=int, default=32768, help="batch size = tokens // T")
p.add_argument("--base", type=float, default=10_000.0)
p.add_argument("--angle", type=float, default=90.0)
args = p.parse_args()


def bench(fn, q, k, v, reps=20):
    for x in (q, k, v):
        x.requires_grad_(True)
    g = torch.randn_like(q)
    for _ in range(3):
        fn(q, k, v).backward(g)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn(q, k, v).backward(g)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


theta = rope.inv_freq(64, args.base)
# the modules cache the thresholds (as patch_llama does); Flash-HiSA with chunked=True takes q / k in the chunked
# layout, as in training (patch_llama(..., chunked=True)): no permutation per call. Random inputs: layout irrelevant.
impls = {"dense (SDPA)": lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True),
         "HiSA (Triton)": HiSAAttention(theta, args.angle, impl="triton"),
         "HiSA block (Triton)": HiSAAttention(theta, args.angle, impl="triton_block", rule_block=128)}
try:
    from hisa.kernels.flash import flash_dense_attention, interface

    interface()
    impls["dense (FlashAttention-3)"] = flash_dense_attention
    impls["Flash-HiSA"] = HiSAAttention(theta, args.angle, impl="flash", chunked=True)
except ImportError:
    pass

print(f"forward + backward, ms per call ({args.tokens} tokens per call, 9 / 3 heads, d = 64, bf16)")
print(f"{'T':>6}  " + "  ".join(f"{n:>24}" for n in impls))
for T in args.T:
    B = max(1, args.tokens // T)
    mk = lambda h: torch.randn(B, h, T, 64, device="cuda", dtype=torch.bfloat16)
    times = [bench(fn, mk(9), mk(3), mk(3)) for fn in impls.values()]
    print(f"{T:>6}  " + "  ".join(f"{t:>24.2f}" for t in times))
