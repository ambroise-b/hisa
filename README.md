# HiSA: Hidden Sparse Attention

Official implementation of **HiSA: Sparsifying the Hidden Dimension of RoPE Attention in LLMs**
([paper](https://arxiv.org/abs/XXXX.XXXXX)).

With rotary position embeddings (RoPE), the attention score between a query and a key is a sum of one part per
frequency pair, and pair *l* is rotated by the angle (*i − j*)·θ<sub>*l*</sub>. HiSA keeps the score part of pair
*l* only while this rotation stays below an angle budget Φ (90° by default):

$$
s_{ij} = \frac{1}{\sqrt{d}} \sum_{l=0}^{d/2-1} \mathbf{1}\big[(i-j)\,\theta_l < \Phi\big]\,
\big\langle R_{i\theta_l} q_i^{(l)},\, R_{j\theta_l} k_j^{(l)} \big\rangle
$$

Fast pairs only see nearby keys and slow pairs see far: each pair has its own window
Δ<sub>max,*l*</sub> = Φ/θ<sub>*l*</sub>, and keys beyond the largest one (the *reach*, 11.8K tokens at RoPE base
10⁴) leave the softmax. The attention is sparse in the **hidden dimension** of the query–key product, not over
tokens; the pattern is static (set by the positions) and adds no parameter. Φ = ∞ is dense attention.

Trained from scratch (SmolLM-135M/360M architectures), HiSA reaches a lower validation loss than dense attention
while computing about a third of the query–key products, and it shrinks the KV cache during decoding.

## What is in this repository

| Implementation | Rule | Hardware | Use |
|---|---|---|---|
| `impl="reference"` | exact or block | any (PyTorch) | the readable definition; tests |
| `impl="triton"` | exact | NVIDIA GPU, fp32 / bf16 | training and inference (forward + backward) |
| `impl="triton_block"` | block | NVIDIA GPU, fp32 / bf16 | portable version of the Flash-HiSA rule |
| `impl="flash"` (Flash-HiSA) | block (128) | Hopper (H100/H200), bf16 | fastest training: the speed of FlashAttention-3 |
| `HiSAKVCache`, `HiSADecoder` | block (128) | Hopper, bf16 | decoding with a ring-buffer KV cache (~⅓ smaller) |

The **exact rule** keeps pair *l* for every query–key distance below its window. The **block rule** (Flash-HiSA)
groups positions into blocks of 128 tokens and decides per block pair, at the blocks' minimum distance: it keeps
every pair of the exact rule plus up to 127 tokens more, and maps onto whole FlashAttention tiles. The two rules are
**different models**: train and run a model with the same one.

Limits: the kernels are written for **head dimension 64** (32 RoPE pairs; the reference takes any even d).
Flash-HiSA needs an sm90 GPU and bf16. Any angle budget and RoPE base work with every implementation; see
[supported settings and limits](docs/usage.md#supported-settings-and-limits). Flash-HiSA lives in a fork of FlashAttention,
[flash-attention-hisa](https://github.com/ambroise-b/flash-attention-hisa), included as the submodule
`third_party/flash-attention` (branch `hisa`, one commit on top of FlashAttention 3.0.1).

## Install

```bash
git clone --recursive https://github.com/ambroise-b/hisa
cd hisa
pip install -e ".[hf]"          # PyTorch + Triton implementations, Hugging Face integration
scripts/build_flash.sh          # optional: Flash-HiSA (sm90, CUDA >= 12.3), about 4 min on 32 cores
```

See [docs/install.md](docs/install.md) for requirements, the Flash-HiSA build and running the tests.

## Quickstart

```python
import torch
from hisa import hisa_attention, rope

B, H, H_kv, T, d = 2, 9, 3, 4096, 64
q = torch.randn(B, H, T, d, device="cuda", dtype=torch.bfloat16)      # after RoPE
k = torch.randn(B, H_kv, T, d, device="cuda", dtype=torch.bfloat16)   # after RoPE (GQA: H_kv divides H)
v = torch.randn(B, H_kv, T, d, device="cuda", dtype=torch.bfloat16)
theta = rope.inv_freq(d, base=10_000.0)                                # the RoPE frequencies of the model

out = hisa_attention(q, k, v, theta, max_angle_deg=90.0, impl="triton")   # exact rule
out = hisa_attention(q, k, v, theta, max_angle_deg=90.0, impl="flash")    # Flash-HiSA (block rule)
```

In a Hugging Face Llama model (e.g. SmolLM), from scratch or from a pretrained checkpoint:

```python
from transformers import LlamaForCausalLM
from hisa.integrations.hf import patch_llama

model = LlamaForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
patch_llama(model, max_angle_deg=90.0, impl="triton")   # every layer's attention becomes HiSA, weights kept
```

A pretrained dense model should be **converted by continued training** (the paper recovers the dense loss after
1.35B tokens); switching the attention without training degrades it. More in [docs/usage.md](docs/usage.md).

## Results (from the paper)

135M parameters, 4K context, RoPE base 10⁴, Φ = 90°, one H100/H200 GPU:

| Attention | Kernel | Val. loss | Tokens/s (training) | Query–key products computed |
|---|---|---|---|---|
| Dense | PyTorch SDPA | 2.906 | 205k | 100% |
| Dense | FlashAttention-3 | 2.904 | 236k | 100% |
| HiSA | Triton (exact rule) | **2.880** | 119k | 29% |
| HiSA, block rule | Triton | 2.888 | 191k | 35% |
| Flash-HiSA | FlashAttention-3 fork | 2.892 | **238k** | 35% |

Decoding at 16K context (b = 2·10⁴), batch 128: KV cache 29.1 GiB instead of 45.0 GiB and 13.0k instead of 9.2k
tokens/s against dense FlashAttention-3 (11.6k for dense attention through the same decode kernel).

## Documentation

- [docs/install.md](docs/install.md): requirements, building Flash-HiSA, tests.
- [docs/usage.md](docs/usage.md): the API, choosing the budget, Hugging Face models, conversion, decoding.
- [docs/kernels.md](docs/kernels.md): how the kernels work (chunks, block rule, ring buffers) and where to find what.
- [examples/](examples): `quickstart.py`, `train_tiny_llama.py`, `bench.py`.

## Repository layout

```
hisa/
├── __init__.py              public API: hisa_attention, HiSAAttention, hisa_attention_reference, rope
├── rope.py                  the rule: frequencies θ_l, windows Δ_max,l, integer thresholds, pair masks, chunk layout
├── reference.py             HiSA in plain PyTorch (any device / dtype): the readable definition and test oracle
├── attention.py             hisa_attention (function) and HiSAAttention (module): dispatch to an implementation
├── kernels/
│   ├── triton_exact.py      impl="triton": exact rule, forward + backward (Triton)
│   ├── triton_block.py      impl="triton_block": block rule, forward + backward (Triton)
│   ├── flash.py             impl="flash": Flash-HiSA wrappers around the FlashAttention-3 fork (training)
│   └── flash_decode.py      HiSAKVCache: ring-buffer KV cache and the Flash-HiSA decode kernel
└── integrations/
    ├── hf.py                patch_llama: HiSA in Hugging Face Llama models (derived from transformers' LlamaAttention)
    └── hf_decode.py         HiSADecoder: token-by-token decoding of a patched model with HiSAKVCache
tests/
├── test_reference.py        rule, masks and reference against hand-computed cases; API checks (CPU)
├── test_triton.py           Triton kernels against the reference, forward and backward (GPU or Triton interpreter)
├── test_flash.py            Flash-HiSA training and decode kernels against the reference (sm90 + fork build)
└── test_hf.py               patch_llama and the decoder on tiny Llama models
examples/
├── quickstart.py            every available implementation against the reference on random inputs
├── train_tiny_llama.py      end-to-end training of a tiny Llama on a copy task (torch.compile included)
└── bench.py                 forward + backward time per call against dense SDPA and FlashAttention-3
docs/                        install.md, usage.md, kernels.md
scripts/build_flash.sh       builds the FlashAttention-3 fork in place (sm90, head dim 64, bf16)
third_party/flash-attention  submodule: the FlashAttention-3 fork with the Flash-HiSA kernels (branch hisa)
```

## Citation

```bibtex
@article{borbely2026hisa,
  title   = {HiSA: Sparsifying the Hidden Dimension of RoPE Attention in LLMs},
  author  = {Borbely, Ambroise and Salzmann, Mathieu and Verscheure, Olivier},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

Apache 2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). The FlashAttention fork keeps FlashAttention's BSD 3-Clause
license.
