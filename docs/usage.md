# Using HiSA

## The function and the module

```python
from hisa import hisa_attention, HiSAAttention, rope

out = hisa_attention(q, k, v, theta, max_angle_deg=90.0, impl="triton", rule_block=None, chunked=False)
attn = HiSAAttention(theta, max_angle_deg=90.0, impl="flash")   # same arguments; out = attn(q, k, v)
```

- `q` [B, H, T, 64] and `k`, `v` [B, H_kv, T, 64]: queries and keys **after RoPE**, causal attention. Grouped-query
  attention: query head h uses key/value head h // (H / H_kv).
- `theta` [32]: the RoPE frequencies that set the windows, normally the model's own (`rope.inv_freq(64, base)`, or
  `model.model.rotary_emb.inv_freq` in Hugging Face).
- `max_angle_deg`: the angle budget Φ. `math.inf` gives dense attention.
- `impl`: `"reference"`, `"triton"` (exact rule), `"triton_block"` or `"flash"` (block rule); see the README table.
- `rule_block`: `None` for the exact rule; the block size S for the block rule (`"flash"`: 128, the default).
- `chunked`: `impl="flash"` only; `q` and `k` are already in the kernels' chunked feature layout (see below).

`HiSAAttention` caches the integer thresholds, so it does no host–device synchronisation per call and works under
`torch.compile`. Under autocast, q/k/v are cast to the autocast dtype (Hugging Face's RoPE returns fp32 q/k).

Useful helpers in `hisa.rope`: `window(theta, angle)` (the window of each pair, Φ/θ<sub>l</sub>),
`reach(theta, angle)` (the distance beyond which keys leave the softmax), `thresholds(theta, angle)` (the integer
windows the kernels use), `pair_masks(theta, angle, T, rule_block)` (which pair is kept for every query–key entry).

## Choosing the budget and the RoPE base

The budget and the base fix every window: Δ<sub>max,l</sub> = Φ · b<sup>2l/d</sup>. In the paper (135M, 2K–4K
context), every budget from 45° to 360° beats dense attention and 90° is the default: about 29% of the query–key
products are computed at 4K. The reach (Φ · b<sup>(d−2)/d</sup>: 11.8K at b = 10⁴, 23K at b = 2·10⁴) should be at
least the context you care about, since keys beyond it are not attended at all; raise the RoPE base for longer
contexts (the paper uses b = 2·10⁴ for 16K).

## Training a Hugging Face Llama model from scratch

```python
from transformers import AutoConfig, LlamaForCausalLM
from hisa.integrations.hf import patch_llama

config = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM-135M")
model = LlamaForCausalLM(config)
patch_llama(model, max_angle_deg=90.0, impl="flash", chunked=True)   # Hopper; or impl="triton" anywhere
model = torch.compile(model.cuda())
# ... your usual training loop, bf16 autocast ...
```

`patch_llama` replaces every layer's attention in place (parameter names are unchanged) and sets
`use_cache=False`: the patched model computes full causal sequences (training, evaluation). For decoding see below.
Padding is not supported: a batch whose `attention_mask` masks any token raises a `ValueError` (the kernels are
causal only). `model.hisa` records the rule; `HiSADecoder` reads it.

`chunked=True` applies RoPE directly in the kernels' feature layout, which saves a permutation copy of q and k in
every layer. It relabels the rows of the q/k projections, so use it **only from scratch** (a model trained with it
must keep it), not with a pretrained Hugging Face checkpoint.

## Converting a pretrained dense model

```python
model = LlamaForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M", dtype=torch.bfloat16)
patch_llama(model, max_angle_deg=90.0, impl="triton")   # chunked=False: keep the checkpoint's layout
# continue training
```

Switching the attention of a trained dense model without training costs some loss (2.906 → 2.937 for the paper's
4K model); continued training recovers it (the paper matches dense continued training after 1.35B tokens).

## Decoding with the ring-buffer KV cache (Flash-HiSA)

For a model with the block rule (trained with `impl="flash"` or `impl="triton_block"` and `rule_block=128`):

```python
from hisa.integrations.hf_decode import HiSADecoder

model = model.to(torch.bfloat16).cuda().eval()
dec = HiSADecoder(model, B=8, S_max=4096)   # rule (angle, frequencies, chunked) from model.hisa
tokens = dec.generate(prompt, max_new_tokens=256)          # prompt: int64 [8, P], equal lengths
losses, seconds = dec.teacher_forced(windows)              # benchmark: CUDA graph, windows int64 [8, S + 1]
print(dec.cache_bytes / 2**30, "GiB")
```

The cache stores each 16-feature chunk of the keys in its own ring buffer, as long as the window of the chunk's
slowest pair (rounded up to 128-token blocks), and the values in a ring as long as the reach: at 4K context and
b = 10⁴, 256 / 256 / 1408 / 4096 tokens for the four key chunks instead of 4096 each. `HiSADecoder` feeds prompts
token by token and all sequences share the same position (no padding); it is a reference loop to build on, not a
serving engine. The low-level cache is `hisa.kernels.flash_decode.HiSAKVCache` (`append`, `attend`).

## Supported settings and limits

**Any angle budget and any RoPE base.** The rule reaches the kernels only as integer windows (and, for Flash-HiSA,
a small table of dropped pairs per block distance) computed on the host, so every implementation accepts any
Φ > 0 (including `math.inf`, dense attention) and any base, with no recompilation. The tests cover bases 30 to
10⁴ and budgets 45°, 90°, 360° and ∞; the paper's models use b = 10⁴ and 2·10⁴.

| | `reference` | `triton`, `triton_block` | `flash` (training) | decoding (`HiSAKVCache`) |
|---|---|---|---|---|
| Hardware | any | NVIDIA GPU | sm90 (H100 / H200) | sm90 |
| dtype | any | fp32, bf16 | bf16 (others are cast) | bf16 |
| Head dimension | any even d | 64 | 64 | 64 |
| Rule block S | any | `triton_block`: multiple of 32 | 128 | 128 |
| Attention | causal | causal, GQA | causal, GQA | one new token per sequence, GQA |
| Not supported | | padding masks, varlen | varlen, paged KV, softcap | paged KV; sequences at different positions; cache length not a multiple of 128 |

**Speed depends on what is dropped.** The kernels skip work in chunks of 8 pairs (16 features). When the windows
are long compared with the context (large budget or base), few chunks are dropped: the result is still exact, but
the speed is close to dense attention. In the paper's setting (90°, b = 10⁴, 4K) about 29% of the query–key
products remain (35% with the block rule).

**Very small windows and the block rule.** The block rule keeps each pair up to S − 1 = 127 tokens beyond its
exact window. When most windows are much shorter than 128 tokens (very small budgets or bases), it is a coarser
approximation of the exact rule; it remains a valid model, as long as training and inference use the same rule.
