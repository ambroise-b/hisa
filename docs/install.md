# Installation

## Requirements

| Part | Needs | Tested with |
|---|---|---|
| `hisa` (reference, Triton kernels) | Python ≥ 3.10, PyTorch ≥ 2.9, Triton ≥ 3.5, an NVIDIA GPU for the Triton kernels | Python 3.11, PyTorch 2.9.1 (CUDA 12.8), Triton 3.5.1, H100 / H200 |
| Hugging Face integration (`[hf]`) | transformers ≥ 5.0 | transformers 5.18.0 |
| Flash-HiSA (`impl="flash"`, decoding) | an sm90 GPU (H100 / H200), CUDA ≥ 12.3, ninja | CUDA 12.8, H100 / H200 |

The reference implementation runs anywhere PyTorch runs (CPU included); it is meant for tests and short sequences.

## Install

```bash
git clone --recursive https://github.com/ambroise-b/hisa
cd hisa
pip install -e ".[hf]"
```

If you cloned without `--recursive`: `git submodule update --init --recursive` (the FlashAttention fork and its
CUTLASS submodule; only needed for Flash-HiSA).

## Building Flash-HiSA

Flash-HiSA is a fork of FlashAttention-3 (`third_party/flash-attention`, branch `hisa`). The build script compiles
only what HiSA needs (sm90, head dimension 64, bf16, forward + backward, split-KV for decoding) in place:

```bash
pip install ninja packaging
scripts/build_flash.sh
```

It takes about 4 minutes on 32 cores with the defaults (`MAX_JOBS=16`, `NVCC_THREADS=2`) and ends with
`Flash-HiSA build OK`. The extension lands in `third_party/flash-attention/hopper/flash_attn_3/`, and
`hisa.kernels.flash` finds it there. If you prefer to install it into your environment, run
`pip install --no-build-isolation .` inside `third_party/flash-attention/hopper` with the same environment variables
as the script (then `import flash_attn_interface` must work from anywhere). To use a build elsewhere, set
`HISA_FLASH_PATH` to its `hopper/` directory.

Common problems:

- **The build is killed** (out of memory): lower `MAX_JOBS`, e.g. `MAX_JOBS=4 scripts/build_flash.sh`.
- **`the installed FlashAttention-3 is not the HiSA fork`**: another FlashAttention-3 (e.g. from PyPI) is found
  first. Uninstall it, or make sure the fork's `hopper/` comes first on `sys.path`.
- **`no kernel image is available`**: the GPU is not sm90. Flash-HiSA only runs on Hopper; use `impl="triton"` or
  `impl="triton_block"` elsewhere.
- **CUDA / PyTorch mismatch**: the CUDA toolkit used by `nvcc` must match the CUDA version of your PyTorch build.

## Running the tests

```bash
pip install -e ".[test]"
pytest tests/test_reference.py tests/test_hf.py            # CPU: the rule, the reference, the HF integration
TRITON_INTERPRET=1 pytest tests/test_triton.py             # CPU: the Triton kernels in the interpreter (slow)
pytest tests                                               # GPU: everything (Flash tests need sm90 + the build)
```

Tests that need hardware or the Flash-HiSA build you do not have are skipped, not failed. On a GPU, the Triton and
Flash-HiSA kernels are checked against the fp64 reference (outputs and gradients), for causality (changing future
tokens never changes past outputs), and the decoder against the full forward pass of a model.
