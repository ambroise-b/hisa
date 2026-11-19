# How the kernels work

## The score as a sum over pairs

With RoPE (Hugging Face layout), features l and l + 32 of a 64-dimensional head form pair l, rotated at frequency
θ<sub>l</sub> = b<sup>−2l/64</sup>. The score splits exactly into one part per pair, and HiSA keeps part l while
(i − j)·θ<sub>l</sub> < Φ, i.e. while i − j < L<sub>l</sub> = ⌈Φ/θ<sub>l</sub>⌉ (integer thresholds computed once in
float64, `hisa.rope.thresholds`, so every implementation applies exactly the same rule). Since the windows grow
with l, the pairs dropped at a given distance are always a **prefix** l < a(distance) of the pairs.

## The chunked feature layout

Tensor cores multiply 16 features at a time (one bf16 K-step), so the kernels permute q and k once so that chunk c
(16 contiguous features) holds pairs 8c … 8c+7 (`hisa.rope.chunk_permutation`). A prefix of pairs is then a prefix
of chunks: skipping the dropped pairs means skipping whole 16-feature slices of Q Kᵀ, plus zeroing the dropped pairs
of the one boundary chunk. Inner products over any subset of pairs are unchanged by the permutation.

## Exact rule (Triton, `hisa/kernels/triton_exact.py`)

FlashAttention-2 structure (query tile × key tiles, online softmax, recomputation in backward). On a tile the
distances span [dmin, dmax]; pairs below a_lo are dropped everywhere (skipped), pairs above a_hi are kept everywhere
(one dot over their chunks), and the few pairs in between are **mixed**: masked element-wise with D < L<sub>l</sub>.
The key loop runs in 4 segments (far → near) that compute 1, 2, 3 or 4 chunks, so the loop bodies have no
data-dependent loads. The result is exact (equal to the reference up to floating-point order).

## Block rule (Triton `triton_block.py`, Flash-HiSA)

Positions are grouped into blocks of S = 128 tokens, and pair l is kept on a whole (query block, key block) pair iff
the blocks' minimum distance dmin = max((bi − bj)·S − (S − 1), 0) is below L<sub>l</sub>. Kernel tiles divide S, so
every tile keeps the same pairs: no mixed pairs, no element-wise masks. It is a slightly denser model than the exact
rule (each pair is kept up to S − 1 tokens longer), so it must be used for training and inference alike.

## Flash-HiSA in the FlashAttention-3 fork

All changes are in one commit on top of FlashAttention 3.0.1 (`third_party/flash-attention`, branch `hisa`; see
`HISA.md` there), as separate compile-time "Sparse" instantiations (hdim 64, bf16, sm90), so the dense kernels are
unchanged:

- `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`: Q Kᵀ sliced per 16-feature chunk; for each key block the chunks below
  K0 are skipped with straight-line GMMA sequences per K0 (no predicated or branched GMMA, which would serialise
  the warpgroup MMAs); the dropped pairs of the boundary chunk are zeroed in shared-memory Q.
- `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp`: the same rule in the backward pass (Sᵀ skips the dropped chunks).
- `hopper/flash_api_stable.cpp`: new op arguments `sparse_a` (the table below), `sparse_rule_block`, and
  `sparse_kc0..3` (decoding).
- The table (`hisa.kernels.flash.table_with_header`): for every block distance nd, the number a[nd] of dropped
  pairs, with a header nd<sub>1..4</sub> (first distance with ≥ 8, 16, 24 dropped pairs, and the number of distances
  with a kept pair) and the thresholds, so the kernel derives the number of skipped chunks per segment without
  memory loads in the main loop.

## Decoding: per-chunk ring buffers (`hisa/kernels/flash_decode.py`, `hopper/sparse_kv.h`)

A cached key needs chunk c only while the slowest pair of the chunk is still kept, and its value only within the
reach. `HiSAKVCache` stores the keys as four ring buffers, one per chunk, of length M<sub>c</sub> = the window of the
chunk's slowest pair rounded up to whole 128-token blocks, and the values in one ring of length M<sub>V</sub> (the
reach, capped at S_max). The token at position t lives in slot t mod M<sub>c</sub> of ring c; a new token overwrites
the slot of a token that has just left that chunk's window, so there is no eviction step. Because rings hold whole
blocks, a 128-key tile never wraps. The decode kernel (one query position per sequence, query heads packed per KV
head, split-KV) reads each key block's kept chunks with `cp.async`; dropped chunks are zero-filled without memory
access. The ring lengths come from the block rule, so the decode output equals the block-rule forward pass.
