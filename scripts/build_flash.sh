#!/usr/bin/env bash
# Build the Flash-HiSA kernels (third_party/flash-attention/hopper, the HiSA fork of FlashAttention-3) in place.
#
# Restricted to what HiSA uses: sm90 (H100 / H200), head dim 64, bf16, forward + backward, causal and local
# windows, GQA, split-KV and varlen (decoding). The extension lands in hopper/flash_attn_3/_C*.so, and
# hisa.kernels.flash finds it there (or wherever `import flash_attn_interface` resolves, if you pip-installed it).
#
#   git submodule update --init --recursive     # the fork and its CUTLASS submodule
#   scripts/build_flash.sh                      # MAX_JOBS / NVCC_THREADS control parallelism (memory!)
#
# Needs: CUDA >= 12.3 (tested 12.8), a PyTorch built for it, ninja, packaging. About 4 min on 32 cores with the
# defaults below (tested with 128 GB of RAM); lower MAX_JOBS if the build is killed for lack of memory.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../third_party/flash-attention/hopper"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
for f in HDIM96 HDIM128 HDIM192 HDIM256 HDIMDIFF64 HDIMDIFF192 FP16 FP8 PAGEDKV APPENDKV SOFTCAP CLUSTER SM80; do
  export "FLASH_ATTENTION_DISABLE_$f=TRUE"
done
export MAX_JOBS="${MAX_JOBS:-16}" NVCC_THREADS="${NVCC_THREADS:-2}"
echo "building in $(pwd) with MAX_JOBS=$MAX_JOBS NVCC_THREADS=$NVCC_THREADS"
time python setup.py build_ext --inplace "$@"
ls -la flash_attn_3/
python -c "import sys; sys.path.insert(0, '.'); import flash_attn_interface, torch; \
assert 'sparse_a' in str(torch.ops.flash_attn_3.fwd.default._schema); print('Flash-HiSA build OK')"
