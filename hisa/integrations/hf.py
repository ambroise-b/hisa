# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Source: transformers 5.18.0, src/transformers/models/llama/modeling_llama.py, class LlamaAttention.
# Modifications (HiSA):
#   - the attention computation after RoPE is a pluggable `core_attention(q, k, v)`
#     (q: [B, H, T, d], k/v: [B, H_kv, T, d], causal, returns [B, H, T, d]);
#   - no KV cache, no padding mask (a mask that masks keys raises): full causal sequences only (training,
#     evaluation; for decoding see hisa.integrations.hf_decode);
#   - `chunked=True` (opt-in, for impl="flash"): the projection outputs are read in the chunked feature layout
#     (16-feature chunks of 8 RoPE pairs) and rotated in it, so the kernels need no permutation. This relabels the
#     q/k projection rows: fine for training from scratch, not compatible with pretrained HF checkpoints.
# Parameter names are unchanged, so pretrained LlamaForCausalLM weights load as is.
"""Use HiSA in a Hugging Face Llama model (LlamaForCausalLM and models with the same attention, e.g. SmolLM).

    model = LlamaForCausalLM(config)              # or LlamaForCausalLM.from_pretrained(...)
    patch_llama(model, max_angle_deg=90.0, impl="triton")
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from transformers import LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from hisa import rope
from hisa.attention import HiSAAttention

CoreAttention = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper, with a pluggable core attention."""

    def __init__(self, config: LlamaConfig, layer_idx: int, core_attention: CoreAttention, chunked: bool = False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        # A plain attribute (not a submodule), so it adds no state_dict keys.
        self.core_attention = core_attention
        self.chunked = chunked

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        if past_key_values is not None:
            raise NotImplementedError("HiSA LlamaAttention has no KV cache (use_cache=False); "
                                      "for decoding see hisa.integrations.hf_decode")
        check_no_padding(attention_mask)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if self.chunked:
            query_states, key_states = rope.apply_rotary_chunked(query_states, key_states, cos, sin)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_output = self.core_attention(query_states, key_states, value_states)  # [B, H, T, d]

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None


def check_no_padding(attention_mask) -> None:
    """Raise if the mask Hugging Face passes to the attention layers masks any key: the HiSA kernels are causal only
    and would attend to padding tokens. Without padding, transformers passes no mask (SDPA path) or, under
    torch.compile, a plain causal mask: its last query row keeps every key. Skipped while compiling (reading the mask
    would break the graph), and for non-tensor masks."""
    if not isinstance(attention_mask, torch.Tensor) or torch.compiler.is_compiling():
        return
    m = attention_mask if attention_mask.dim() == 2 else attention_mask[..., -1, :]  # [B, T] or last query row
    keep = m.bool() if m.dtype == torch.bool or attention_mask.dim() == 2 else m == 0  # additive float: 0 = keep
    if not bool(keep.all()):
        raise ValueError("HiSA attention does not support padding masks (the kernels are causal only): use "
                         "unpadded, equal-length sequences")


def patch_llama(model: LlamaForCausalLM, max_angle_deg: float = 90.0, impl: str = "triton",
                rule_block: int | None = None, chunked: bool = False,
                rule_theta: torch.Tensor | None = None) -> HiSAAttention:
    """Replace every layer's self_attn of `model` in place by HiSA (weights kept); returns the shared core.

    impl / rule_block / chunked: see hisa.hisa_attention. rule_theta: the frequencies that set the windows
    (default: the model's own RoPE frequencies). The model must be used without KV cache (use_cache=False is set)."""
    theta = model.model.rotary_emb.inv_freq if rule_theta is None else rule_theta
    if impl == "flash" and rule_block is None:
        rule_block = 128  # as hisa_attention: recorded in model.hisa (the decoder checks it)
    core = HiSAAttention(theta, max_angle_deg, impl, rule_block, chunked)
    for layer in model.model.layers:
        old = layer.self_attn
        new = LlamaAttention(model.config, old.layer_idx, core_attention=core, chunked=chunked)
        new.load_state_dict(old.state_dict())
        new.to(device=old.q_proj.weight.device, dtype=old.q_proj.weight.dtype)
        layer.self_attn = new
    model.config.use_cache = False
    model.hisa = {"max_angle_deg": max_angle_deg, "impl": impl, "rule_block": rule_block, "chunked": chunked,
                  "rule_theta": theta.detach().float().cpu()}
    return core
