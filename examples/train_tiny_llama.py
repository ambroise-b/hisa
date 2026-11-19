"""Train a tiny Llama (2 layers) with HiSA on a copy task: checks that everything runs end to end (forward,
backward, torch.compile) and that the model learns to attend back T/2 tokens. Each sequence is T/2 random tokens
repeated once; the loss is measured on the copy (random guessing: 5.545).

    python examples/train_tiny_llama.py --impl triton        # any NVIDIA GPU, about a minute
    python examples/train_tiny_llama.py --impl flash         # Hopper + Flash-HiSA build
    python examples/train_tiny_llama.py --angle inf          # dense attention, for comparison

With the defaults (T = 256, 1000 steps), H100, 2026-11-19: the exact rule (--impl triton) reaches a copy loss of
0.03; dense attention learns this toy task faster (0.002 at step 100). At T = 256 the block rule with 128-token blocks
keeps every pair (all keys are in the query's block or the previous one), so --impl triton_block / flash train like
dense attention here: for these kernels the run checks the plumbing, not the sparsity. Longer copy distances need
more steps with HiSA.
"""

import argparse

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from hisa.integrations.hf import patch_llama

p = argparse.ArgumentParser()
p.add_argument("--impl", default="triton", choices=["triton", "triton_block", "flash"])
p.add_argument("--angle", type=float, default=90.0)
p.add_argument("--steps", type=int, default=1000)
p.add_argument("--T", type=int, default=256)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--layers", type=int, default=2)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--warmup", type=int, default=50, help="linear warmup steps")
args = p.parse_args()

cfg = LlamaConfig(vocab_size=256, hidden_size=576, intermediate_size=1536, num_hidden_layers=args.layers,
                  num_attention_heads=9, num_key_value_heads=3, head_dim=64, max_position_embeddings=args.T,
                  use_cache=False)
torch.manual_seed(0)
model = LlamaForCausalLM(cfg)
rule_block = 128 if args.impl != "triton" else None
patch_llama(model, args.angle, impl=args.impl, rule_block=rule_block, chunked=args.impl == "flash")
model = torch.compile(model.cuda())
opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup)))

for step in range(args.steps):
    # second half of every sequence repeats the first half: solvable only by attending back T/2 tokens
    half = torch.randint(0, 256, (args.batch, args.T // 2), device="cuda")
    ids = torch.cat([half, half], 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=ids).logits
    loss = torch.nn.functional.cross_entropy(logits[:, args.T // 2:-1].float().flatten(0, 1), ids[:, args.T // 2 + 1:].flatten())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    sched.step()
    if step % 50 == 0 or step == args.steps - 1:
        print(f"step {step:4d}  copy loss {loss.item():.3f}  (random guessing: {torch.log(torch.tensor(256.)):.3f})")
