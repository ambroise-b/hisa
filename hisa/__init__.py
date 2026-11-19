"""HiSA: Hidden Sparse Attention.

With RoPE, the query-key score is a sum of one part per frequency pair. HiSA keeps the part of pair p only while
the pair's relative rotation (i - j) * theta_p stays below an angle budget (90 degrees by default): fast pairs only
see nearby keys, slow pairs see far. The sparsity is in the hidden dimension of the dot product, static (set by
the positions) and parameter-free.
"""

from hisa import rope
from hisa.attention import IMPLS, HiSAAttention, hisa_attention
from hisa.reference import hisa_attention_reference, sdpa_attention

__version__ = "0.1.0"

__all__ = ["IMPLS", "HiSAAttention", "hisa_attention", "hisa_attention_reference", "rope", "sdpa_attention"]
