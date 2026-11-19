"""Token-by-token decoding of a HiSA Llama model with the Flash-HiSA ring-buffer KV cache (Hopper, bf16).

One decode step = embedding -> per layer (RMSNorm, q/k/v projections, RoPE at position t, cache append, attention
over the cache, o_proj, residual, RMSNorm, MLP, residual) -> norm -> lm_head, written out explicitly with the
model's own modules and weights (as transformers' LlamaDecoderLayer), so that the KV cache is ours.

All sequences of the batch are at the same position t (equal-length prompts, fed token by token). The position
lives in a device tensor and the step only uses static shapes, so it can be captured as a CUDA graph
(`teacher_forced` does this: it is how the paper measures decoding throughput and checks the decoded loss).

The model must use the block rule with rule_block=128 (a model trained with impl="flash" or "triton_block"): the
decode kernel implements that rule. max_angle_deg=inf runs the same kernel dense. For a model patched with
`patch_llama`, the rule (angle, frequencies, chunked layout) is read from `model.hisa` and checked.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from hisa import rope
from hisa.kernels.flash_decode import HiSAKVCache


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def decode_rule(model, max_angle_deg: float | None, rule_theta: torch.Tensor | None, chunked: bool | None):
    """(max_angle_deg, rule_theta, chunked) to decode `model` with. A model patched with patch_llama records its rule
    in `model.hisa`: arguments left to None are taken from it, given ones must agree with it, and its rule must be the
    block rule with rule_block=128 (or dense). Without `model.hisa`: the arguments, defaults 90 degrees, the model's
    RoPE frequencies, HF layout."""
    rule = getattr(model, "hisa", None)
    if rule is None:
        return 90.0 if max_angle_deg is None else max_angle_deg, rule_theta, bool(chunked)
    if rule["rule_block"] != 128 and math.isfinite(rule["max_angle_deg"]):
        raise ValueError(f"the decode kernel implements the block rule with rule_block=128; this model was patched "
                         f"with impl={rule['impl']!r}, rule_block={rule['rule_block']} (a different model)")
    for name, given in (("max_angle_deg", max_angle_deg), ("chunked", chunked)):
        if given is not None and given != rule[name]:
            raise ValueError(f"{name}={given} does not match the model's rule ({name}={rule[name]}, model.hisa)")
    if rule_theta is not None and not torch.equal(rule_theta.detach().float().cpu(), rule["rule_theta"]):
        raise ValueError("rule_theta does not match the model's rule (model.hisa['rule_theta'])")
    return rule["max_angle_deg"], rule["rule_theta"], rule["chunked"]


class HiSADecoder:
    def __init__(self, model, B: int, S_max: int, max_angle_deg: float | None = None,
                 rule_theta: torch.Tensor | None = None, chunked: bool | None = None, num_splits: int = 0):
        """model: LlamaForCausalLM in bf16, on the GPU, eval (patched or not: only its weights are used).
        max_angle_deg, rule_theta (the frequencies of the rule), chunked (q / k rotated in the chunked layout): None =
        from model.hisa if the model was patched with patch_llama, else 90 degrees, the model's RoPE frequencies and
        the HF layout; see decode_rule. S_max: a multiple of 128."""
        max_angle_deg, rule_theta, chunked = decode_rule(model, max_angle_deg, rule_theta, chunked)
        self.model, self.B, self.S_max, self.chunked = model, B, S_max, chunked
        cfg = model.config
        p = next(model.parameters())
        self.device, self.dtype = p.device, p.dtype
        self.H, self.Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.d = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        pos = torch.arange(S_max, device=self.device)[None]
        dummy = torch.zeros(1, device=self.device, dtype=self.dtype)
        cos, sin = model.model.rotary_emb(dummy, pos)               # [1, S, d] in the model dtype (as HF)
        cos, sin = cos[0], sin[0]
        if chunked:
            perm = rope.chunk_permutation(self.d).to(self.device)
            cos, sin = cos[:, perm], sin[:, perm]
        self.cos, self.sin = cos.contiguous(), sin.contiguous()
        theta = model.model.rotary_emb.inv_freq if rule_theta is None else rule_theta
        self.cache = HiSAKVCache(cfg.num_hidden_layers, B, self.H, self.Hkv, S_max, theta, max_angle_deg,
                                 self.device, self.dtype, num_splits)
        self.t = torch.zeros(1, device=self.device, dtype=torch.int32)
        self.graph = None

    @property
    def cache_bytes(self) -> int:
        return self.cache.nbytes

    def reset(self) -> None:
        self.t.zero_()  # the rings are overwritten in order; stale slots are never read (masked by t)

    def _attention(self, li: int, attn, h: torch.Tensor) -> torch.Tensor:
        B, H, Hkv, d = self.B, self.H, self.Hkv, self.d
        q = attn.q_proj(h).view(B, H, d)
        k = attn.k_proj(h).view(B, Hkv, d)
        v = attn.v_proj(h).view(B, Hkv, d)
        cos = self.cos.index_select(0, self.t)                      # [1, d]
        sin = self.sin.index_select(0, self.t)
        rot = rope.rotate_half_chunked if self.chunked else _rotate_half
        q = q * cos + rot(q) * sin
        k = k * cos + rot(k) * sin
        to_chunks = (lambda x: x) if self.chunked else self.cache.to_chunked
        self.cache.append(li, to_chunks(k), v, self.t)
        o = self.cache.attend(li, to_chunks(q), self.t)             # [B, H, d]
        return attn.o_proj(o.reshape(B, H * d))

    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [B] (int64, at position self.t) -> logits [B, vocab] (fp32). Does not advance t."""
        m = self.model.model
        h = m.embed_tokens(tokens)                                  # [B, hidden]
        for li, layer in enumerate(m.layers):
            h = h + self._attention(li, layer.self_attn, layer.input_layernorm(h))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        return self.model.lm_head(m.norm(h)).float()

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Greedy generation (eager, no CUDA graph). prompt int64 [B, P] (same length for all sequences)
        -> [B, P + max_new_tokens]."""
        self.reset()
        out = [prompt]
        logits = None
        for i in range(prompt.shape[1]):                            # prefill, token by token
            logits = self.step(prompt[:, i])
            self.t.add_(1)
        for _ in range(max_new_tokens):
            nxt = logits.argmax(-1)
            out.append(nxt[:, None])
            logits = self.step(nxt)
            self.t.add_(1)
        return torch.cat(out, dim=1)

    # ------------------------------------------------------------------ teacher-forced decoding (benchmark)

    def _graph_body(self) -> None:
        tok = self.inputs.index_select(1, self.t.long())[:, 0]
        tgt = self.targets.index_select(1, self.t.long())[:, 0]
        logits = self.step(tok)
        ce = F.cross_entropy(logits, tgt, reduction="none")       # [B]
        self.loss.index_copy_(1, self.t.long(), ce[:, None])
        self.t.add_(1)

    def capture(self) -> None:
        """Record one teacher-forced step (read token t, write loss[:, t], t += 1) as a CUDA graph."""
        self.inputs = torch.zeros(self.B, self.S_max, device=self.device, dtype=torch.long)
        self.targets = torch.zeros_like(self.inputs)
        self.loss = torch.zeros(self.B, self.S_max, device=self.device, dtype=torch.float32)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):                                  # warmup (Triton compilation, FA3 init)
            for _ in range(3):
                self.reset()
                self._graph_body()
        torch.cuda.current_stream().wait_stream(s)
        self.reset()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._graph_body()
        self.reset()

    @torch.no_grad()
    def teacher_forced(self, windows: torch.Tensor) -> tuple[torch.Tensor, float]:
        """windows int64 [B, S + 1] (S <= S_max) -> (per-position loss [B, S], seconds for the S steps)."""
        S = windows.shape[1] - 1
        if self.graph is None:
            self.capture()
        self.inputs[:, :S].copy_(windows[:, :S])
        self.targets[:, :S].copy_(windows[:, 1:])
        self.reset()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(S):
            self.graph.replay()
        end.record()
        torch.cuda.synchronize()
        return self.loss[:, :S].clone(), start.elapsed_time(end) / 1e3
