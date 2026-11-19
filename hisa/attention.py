"""User-facing API: one function and one module for every implementation of HiSA.

    out = hisa_attention(q, k, v, theta, max_angle_deg=90.0, impl="triton")

q [B, H, T, 64], k, v [B, H_kv, T, 64]: queries and keys AFTER RoPE (Hugging Face pair layout, unless
chunked=True), causal. theta: the RoPE frequencies ([32], e.g. hisa.rope.inv_freq(64, base) or a Hugging Face
model's `rotary_emb.inv_freq`).

impl          rule        hardware / dtypes                          training
"reference"   exact/block any (PyTorch), any dtype                   autograd; memory O(T^2 P): tests only
"triton"      exact       NVIDIA GPU (Triton), fp32 / bf16     forward + backward
"triton_block" block      NVIDIA GPU (Triton), fp32 / bf16     forward + backward
"flash"       block       Hopper (sm90), bf16, FlashAttention-3 fork  forward + backward (fastest)

The exact rule and the block rule are different models: train and run a model with the same rule (and, for the
block rule, the same rule_block). max_angle_deg = inf gives dense attention for every impl.
"""

from __future__ import annotations

import math

import torch

from hisa import rope
from hisa.reference import hisa_attention_reference

IMPLS = ("reference", "triton", "triton_block", "flash")


def _check_impl(impl: str, rule_block: int | None, chunked: bool) -> None:
    if impl not in IMPLS:
        raise ValueError(f"unknown impl {impl!r}, expected one of {IMPLS}")
    if impl == "triton" and rule_block is not None:
        raise ValueError("impl='triton' is the exact rule (rule_block=None); use 'triton_block' or 'flash'")
    if impl in ("triton_block", "flash") and rule_block is None:
        raise ValueError(f"impl={impl!r} is the block rule: pass rule_block (128 for 'flash')")
    if impl == "triton_block" and not (rule_block >= 32 and rule_block % 32 == 0):
        raise ValueError(f"impl='triton_block' needs rule_block a positive multiple of 32 (got {rule_block})")
    if impl == "flash" and rule_block != 128:
        raise ValueError("the Flash-HiSA kernels are compiled for rule_block=128")
    if chunked and impl != "flash":
        raise ValueError("chunked=True (q / k in the chunked feature layout) is only supported by impl='flash'")


def hisa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    theta: torch.Tensor,
    max_angle_deg: float = 90.0,
    impl: str = "triton",
    rule_block: int | None = None,
    chunked: bool = False,
) -> torch.Tensor:
    """Causal HiSA attention -> [B, H, T, d]. See the module docstring for the arguments.
    rule_block: None = exact rule; S = block rule with S-token blocks (default for impl='flash': 128)."""
    if impl == "flash" and rule_block is None:
        rule_block = 128
    _check_impl(impl, rule_block, chunked)
    if impl == "reference":
        return hisa_attention_reference(q, k, v, theta, max_angle_deg, rule_block)
    L = rope.thresholds(theta.cpu(), max_angle_deg)
    if impl == "flash":
        from hisa.kernels.flash import hisa_flash

        return hisa_flash(q, k, v, L.tolist(), rule_block, chunked)
    if impl == "triton_block":
        from hisa.kernels.triton_block import hisa_triton_block

        return hisa_triton_block(q, k, v, L.to(q.device), rule_block)
    from hisa.kernels.triton_exact import hisa_triton_exact

    return hisa_triton_exact(q, k, v, L.to(q.device))


class HiSAAttention(torch.nn.Module):
    """HiSA as a module: (q, k, v) -> out, with the rule fixed at construction (thresholds cached, no host-device
    sync per call, safe under torch.compile). Same arguments as `hisa_attention`."""

    def __init__(self, theta: torch.Tensor, max_angle_deg: float = 90.0, impl: str = "triton",
                 rule_block: int | None = None, chunked: bool = False):
        super().__init__()
        if impl == "flash" and rule_block is None:
            rule_block = 128
        _check_impl(impl, rule_block, chunked)
        self.max_angle_deg, self.impl, self.rule_block, self.chunked = max_angle_deg, impl, rule_block, chunked
        self.register_buffer("theta", theta.detach().float().cpu().clone(), persistent=False)
        self._L_cpu = rope.thresholds(self.theta.cpu(), max_angle_deg)
        self._L_list = self._L_cpu.tolist()
        self._L_dev: dict = {}

    @property
    def reach(self) -> float:
        """Distance beyond which a key is outside the softmax (math.inf for dense attention)."""
        return rope.reach(self.theta, self.max_angle_deg)

    def _L(self, device) -> torch.Tensor:
        key = str(device)
        if key not in self._L_dev:
            self._L_dev[key] = self._L_cpu.to(device)
        return self._L_dev[key]

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.impl == "reference":
            return hisa_attention_reference(q, k, v, self.theta.to(q.device), self.max_angle_deg, self.rule_block)
        if self.impl == "flash":
            from hisa.kernels.flash import hisa_flash

            return hisa_flash(q, k, v, self._L_list, self.rule_block, self.chunked)
        if self.impl == "triton_block":
            from hisa.kernels.triton_block import hisa_triton_block

            return hisa_triton_block(q, k, v, self._L(q.device), self.rule_block)
        from hisa.kernels.triton_exact import hisa_triton_exact

        return hisa_triton_exact(q, k, v, self._L(q.device))

    def extra_repr(self) -> str:
        angle = "inf" if math.isinf(self.max_angle_deg) else f"{self.max_angle_deg:g}"
        return f"max_angle_deg={angle}, impl={self.impl!r}, rule_block={self.rule_block}, chunked={self.chunked}"
