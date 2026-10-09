"""Map causal memory tokens to FastWAM action hidden-space residuals."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class BehaviorPrefixAdapterConfig:
    memory_dim: int = 256
    action_horizon: int = 32
    action_hidden_dim: int = 1024
    num_heads: int = 8
    mlp_ratio: float = 2.0
    gate_init: float = 0.0

    def __post_init__(self) -> None:
        if min(self.memory_dim, self.action_horizon, self.action_hidden_dim, self.num_heads) < 1:
            raise ValueError("adapter dimensions and action horizon must be positive")
        if self.action_hidden_dim % self.num_heads:
            raise ValueError("action_hidden_dim must be divisible by num_heads")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")


class BehaviorPrefixAdapter(nn.Module):
    def __init__(self, config: BehaviorPrefixAdapterConfig | None = None) -> None:
        super().__init__(); self.config = config or BehaviorPrefixAdapterConfig(); cfg = self.config
        self.memory_norm = nn.LayerNorm(cfg.memory_dim)
        self.memory_project = nn.Linear(cfg.memory_dim, cfg.action_hidden_dim)
        self.behavior_queries = nn.Parameter(torch.randn(1, cfg.action_horizon, cfg.action_hidden_dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(cfg.action_hidden_dim, cfg.num_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(cfg.action_hidden_dim), nn.Linear(cfg.action_hidden_dim, int(cfg.mlp_ratio * cfg.action_hidden_dim)), nn.GELU(), nn.Linear(int(cfg.mlp_ratio * cfg.action_hidden_dim), cfg.action_hidden_dim))
        self.output = nn.Linear(cfg.action_hidden_dim, cfg.action_hidden_dim)
        # Match Zeva's stable initialization: the residual projector is
        # expressive from the first step, while tanh(gate_init)=0 keeps the
        # complete addon an exact no-op until the scalar gate moves.
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.pim_gate = nn.Parameter(torch.tensor(float(cfg.gate_init)))

    def forward(self, memory_tokens: Tensor, memory_mask: Tensor, action_horizon: int | None = None) -> Tensor:
        cfg = self.config; horizon = cfg.action_horizon if action_horizon is None else int(action_horizon)
        if memory_tokens.ndim != 3 or memory_tokens.shape[0:2] != memory_mask.shape or memory_tokens.shape[-1] != cfg.memory_dim:
            raise ValueError("memory_tokens/memory_mask shape mismatch")
        if horizon != cfg.action_horizon:
            raise ValueError(f"V1 action horizon is fixed at {cfg.action_horizon}")
        memory_tokens = memory_tokens.to(
            device=self.memory_project.weight.device,
            dtype=self.memory_project.weight.dtype,
        )
        memory_mask = memory_mask.to(device=memory_tokens.device, dtype=torch.bool)
        queries = self.behavior_queries.expand(memory_tokens.shape[0], -1, -1)
        keys = self.memory_project(self.memory_norm(memory_tokens))
        padding = ~memory_mask.bool(); empty = ~memory_mask.any(dim=-1); padding = padding.clone(); padding[empty, 0] = False
        attended, _ = self.cross_attention(queries, keys, keys, key_padding_mask=padding, need_weights=False)
        return self.output(attended + self.mlp(attended))

    def gated(self, memory_tokens: Tensor, memory_mask: Tensor, action_horizon: int | None = None) -> Tensor:
        return torch.tanh(self.pim_gate) * self.forward(memory_tokens, memory_mask, action_horizon)


@dataclass(frozen=True)
class ExactZevaPolicyInjectionConfig:
    """FastWAM dimensions for the original Zeva two-branch injection path."""

    memory_dim: int = 256
    context_dim: int = 4096
    global_dim: int = 256
    phase_dim: int = 128
    effect_dim: int = 128
    effect_history_length: int = 4
    action_dim: int = 14
    action_horizon: int = 32
    prior_hidden_dim: int = 256
    prior_num_heads: int = 4
    action_hidden_dim: int = 1024
    leading_condition_steps: int = 0
    prior_loss_weight: float = 0.01
    prior_dropout_rate: float = 0.4
    prior_inference_guidance_scale: float = 0.5
    prompt_gate_init: float = 0.0
    min_std: float = 1.0e-3

    def __post_init__(self) -> None:
        values = (
            self.memory_dim, self.context_dim, self.global_dim, self.phase_dim,
            self.effect_dim, self.effect_history_length, self.action_dim,
            self.action_horizon, self.prior_hidden_dim, self.prior_num_heads,
            self.action_hidden_dim,
        )
        if min(values) < 1:
            raise ValueError("Zeva policy-injection dimensions must be positive")
        if self.prior_hidden_dim % self.prior_num_heads:
            raise ValueError("prior_hidden_dim must be divisible by prior_num_heads")
        if not 0 <= self.leading_condition_steps <= self.action_horizon:
            raise ValueError("leading_condition_steps must be within action_horizon")
        if self.prior_loss_weight < 0 or self.prior_dropout_rate < 0 or self.prior_dropout_rate > 1:
            raise ValueError("invalid Zeva prior loss/dropout configuration")
        if self.min_std <= 0:
            raise ValueError("min_std must be positive")


class FastWAMPolicyInjectionPrior(nn.Module):
    """Zeva's action prior, adapted only for FastWAM's action dimensions."""

    def __init__(self, config: ExactZevaPolicyInjectionConfig) -> None:
        super().__init__()
        self.config = config
        self.global_to_anchors = nn.Linear(
            config.global_dim, 8 * config.prior_hidden_dim
        )
        self.anchor_position = nn.Parameter(
            torch.randn(1, 8, config.prior_hidden_dim) * 0.02
        )
        self.phase_query = nn.Sequential(
            nn.LayerNorm(config.phase_dim), nn.Linear(config.phase_dim, config.prior_hidden_dim)
        )
        self.effect_project = nn.Sequential(
            nn.LayerNorm(config.effect_dim), nn.Linear(config.effect_dim, config.prior_hidden_dim)
        )
        self.effect_position = nn.Parameter(
            torch.randn(1, config.effect_history_length, config.prior_hidden_dim) * 0.02
        )
        self.bos_effect = nn.Parameter(
            torch.randn(1, 1, config.effect_dim) * 0.02
        )
        self.effect_attention = nn.MultiheadAttention(
            config.prior_hidden_dim, config.prior_num_heads, batch_first=True
        )
        self.effect_norm = nn.LayerNorm(config.prior_hidden_dim)
        self.effect_gate = nn.Sequential(
            nn.LayerNorm(2 * config.prior_hidden_dim),
            nn.Linear(2 * config.prior_hidden_dim, config.prior_hidden_dim),
            nn.Sigmoid(),
        )
        self.progress_attention = nn.MultiheadAttention(
            config.prior_hidden_dim, config.prior_num_heads, batch_first=True
        )
        self.context_norm = nn.LayerNorm(config.prior_hidden_dim)
        self.distribution_head = nn.Sequential(
            nn.Linear(config.prior_hidden_dim, config.prior_hidden_dim),
            nn.GELU(),
            nn.Linear(config.prior_hidden_dim, 2 * config.action_horizon * config.action_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.anchor_position, std=0.02)
        nn.init.normal_(self.effect_position, std=0.02)
        nn.init.normal_(self.bos_effect, std=0.02)

    def forward(
        self,
        task_tokens: Tensor,
        phase: Tensor,
        bit_effects: Tensor,
        bit_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.config
        batch = task_tokens.shape[0]
        expected = {
            "task_tokens": (batch, cfg.global_dim),
            "phase": (batch, cfg.phase_dim),
            "bit_effects": (batch, cfg.effect_history_length, cfg.effect_dim),
            "bit_mask": (batch, cfg.effect_history_length),
        }
        for name, shape in expected.items():
            if tuple(locals()[name].shape) != shape:
                raise ValueError(f"Expected {name} {shape}, got {tuple(locals()[name].shape)}")
        anchors = self.global_to_anchors(task_tokens).view(batch, 8, cfg.prior_hidden_dim)
        anchors = anchors + self.anchor_position
        phase_query = self.phase_query(phase)
        bit_mask = bit_mask.to(torch.bool)
        bit_input = torch.where(
            bit_mask.unsqueeze(-1),
            bit_effects,
            self.bos_effect.expand(batch, cfg.effect_history_length, -1),
        )
        bit = self.effect_project(bit_input) + self.effect_position
        effect_context, _ = self.effect_attention(
            phase_query[:, None], bit, bit, need_weights=False
        )
        effect_context = self.effect_norm(effect_context[:, 0])
        gate = self.effect_gate(torch.cat((phase_query, effect_context), dim=-1))
        local_query = gate * phase_query + (1.0 - gate) * effect_context
        context, _ = self.progress_attention(local_query[:, None], anchors, anchors, need_weights=False)
        params = self.distribution_head(self.context_norm(context[:, 0]))
        mean, raw_scale = params.chunk(2, dim=-1)
        mean = mean.view(batch, cfg.action_horizon, cfg.action_dim)
        std = (F.softplus(raw_scale) + cfg.min_std).view(batch, cfg.action_horizon, cfg.action_dim)
        return mean, std


def gaussian_prior_nll(target: Tensor, mean: Tensor, std: Tensor, valid: Tensor | None = None) -> Tensor:
    """Zeva's diagonal-Gaussian action-prior objective."""
    if target.shape != mean.shape or mean.shape != std.shape:
        raise ValueError("target, mean, and std must have the same shape")
    nll = 0.5 * (((target - mean) / std).square() + 2.0 * std.log()).mean(dim=-1)
    if valid is None:
        return nll.mean()
    if valid.shape != nll.shape:
        raise ValueError("valid must have shape [B,H]")
    valid = valid.to(device=nll.device, dtype=nll.dtype)
    return (nll * valid).sum() / valid.sum().clamp_min(1.0)


class ExactZevaPolicyInjectionAdapter(nn.Module):
    """Separate Zeva behavior-prefix and action-prior injection branches.

    The prefix branch mirrors Zeva's policy-side layout: a task-conditioned
    base behavior token is written to a reserved slot, and the Causal Prompt
    contributes a gated residual to that same slot.  The action-prior branch
    remains independent and is projected into ActionDiT hidden space.
    """

    is_exact_zeva = True

    def __init__(self, config: ExactZevaPolicyInjectionConfig | None = None) -> None:
        super().__init__()
        self.config = config or ExactZevaPolicyInjectionConfig()
        cfg = self.config
        self.prior = FastWAMPolicyInjectionPrior(cfg)
        self.action_prior_adapter = nn.Linear(cfg.action_dim, cfg.action_hidden_dim)
        self.behavior_global_projector = nn.Linear(cfg.global_dim, cfg.context_dim)
        self.prefix_project = nn.Linear(cfg.memory_dim, cfg.context_dim)
        self.pim_gate = nn.Parameter(torch.tensor(float(cfg.prompt_gate_init)))
        nn.init.zeros_(self.action_prior_adapter.weight)
        nn.init.zeros_(self.action_prior_adapter.bias)
        nn.init.xavier_uniform_(self.behavior_global_projector.weight)
        nn.init.zeros_(self.behavior_global_projector.bias)
        nn.init.xavier_uniform_(self.prefix_project.weight)
        nn.init.zeros_(self.prefix_project.bias)

    def policy_injection_parameters(self):
        yield from self.prior.parameters()
        yield from self.action_prior_adapter.parameters()
        yield from self.behavior_global_projector.parameters()

    def pim_adapter_parameters(self):
        yield from self.prefix_project.parameters()
        yield self.pim_gate

    @torch.no_grad()
    def reset_pim_parameters(self) -> None:
        nn.init.xavier_uniform_(self.prefix_project.weight)
        nn.init.zeros_(self.prefix_project.bias)
        self.pim_gate.fill_(float(self.config.prompt_gate_init))

    def causal_prompt_prefix(
        self,
        causal_prompt: Tensor,
        pim_mask: Tensor,
        *,
        enable_pim: bool = True,
    ) -> Tensor:
        if causal_prompt.ndim != 2:
            raise ValueError("causal_prompt must be [B,D]")
        if causal_prompt.shape[-1] != self.config.memory_dim:
            raise ValueError(
                f"causal_prompt last dim must be {self.config.memory_dim}"
            )
        if pim_mask.ndim != 2 or pim_mask.shape[0] != causal_prompt.shape[0]:
            raise ValueError("pim_mask must be [B,K]")
        prompt = self.prefix_project(
            causal_prompt.to(
                device=self.prefix_project.weight.device,
                dtype=self.prefix_project.weight.dtype,
            )
        )[:, None]
        if not enable_pim:
            return torch.zeros_like(prompt)
        has_pim = (
            pim_mask.any(dim=-1, keepdim=True)
            .to(device=prompt.device, dtype=prompt.dtype)
            .unsqueeze(-1)
        )
        return torch.tanh(self.pim_gate).to(prompt.dtype) * prompt * has_pim

    def behavior_prefix_slot(
        self,
        task_tokens: Tensor,
        causal_prompt: Tensor,
        pim_mask: Tensor,
        *,
        enable_pim: bool = True,
    ) -> Tensor:
        """Build the single token written to the reserved behavior slot."""
        if task_tokens.ndim != 2 or task_tokens.shape[-1] != self.config.global_dim:
            raise ValueError(
                "task_tokens shape does not match the exact Zeva adapter: "
                f"expected [B,{self.config.global_dim}], got {tuple(task_tokens.shape)}"
            )
        if task_tokens.shape[0] != causal_prompt.shape[0]:
            raise ValueError("task_tokens and causal_prompt must have the same batch size")
        base_prefix = self.behavior_global_projector(
            task_tokens.to(
                device=self.behavior_global_projector.weight.device,
                dtype=self.behavior_global_projector.weight.dtype,
            )
        )[:, None]
        pim_delta = self.causal_prompt_prefix(
            causal_prompt,
            pim_mask,
            enable_pim=enable_pim,
        )
        return base_prefix + pim_delta.to(
            device=base_prefix.device,
            dtype=base_prefix.dtype,
        )

    def action_prior_residual(self, prior_mean: Tensor, *, training: bool) -> Tensor:
        residual = self.action_prior_adapter(prior_mean)
        if training and self.config.prior_dropout_rate:
            keep = torch.rand((residual.shape[0], 1, 1), device=residual.device) >= self.config.prior_dropout_rate
            residual = residual * keep.to(residual.dtype)
        elif not training:
            residual = residual * self.config.prior_inference_guidance_scale
        if self.config.leading_condition_steps:
            residual = torch.cat(
                (
                    residual.new_zeros((residual.shape[0], self.config.leading_condition_steps, residual.shape[-1])),
                    residual,
                ),
                dim=1,
            )
        return residual

    def add_prefix_to_context(self, context: Tensor, prefix: Tensor) -> Tensor:
        """Backward-compatible legacy helper.

        The exact Zeva path must use :meth:`prepend_behavior_prefix_slot`,
        which keeps the instruction sequence untouched.  This method remains
        available for old callers that explicitly request the former residual
        behavior, but is intentionally no longer used by the exact path.
        """
        if context.ndim != 3 or prefix.shape != (context.shape[0], 1, context.shape[-1]):
            raise ValueError("context/prefix shapes do not match")
        result = context.clone()
        result[:, :1] = result[:, :1] + prefix.to(device=context.device, dtype=context.dtype)
        return result

    def prepend_behavior_prefix_slot(
        self,
        context: Tensor,
        context_mask: Tensor,
        causal_prompt: Tensor,
        pim_mask: Tensor,
        task_tokens: Tensor,
        *,
        enable_pim: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Prepend one independent Causal-Prompt slot to FastWAM context.

        Zeva reserves a learned behavior slot before the language prefix in its
        packed sequence.  FastWAM has separate video/action experts whose
        conditioning enters through cross-attention, so the equivalent slot is
        a dedicated context token before the original text tokens.  The slot
        contains the task base prefix plus the gated Causal Prompt residual;
        the original text tensor and mask are copied unchanged after it.
        """
        if context.ndim != 3 or context_mask.shape != context.shape[:2]:
            raise ValueError("context/context_mask shapes do not match")
        if causal_prompt.ndim != 2 or causal_prompt.shape[0] != context.shape[0]:
            raise ValueError("causal_prompt must have the same batch size as context")
        prefix = self.behavior_prefix_slot(
            task_tokens,
            causal_prompt,
            pim_mask,
            enable_pim=enable_pim,
        )
        if prefix.shape != (context.shape[0], 1, context.shape[-1]):
            raise ValueError(
                "Causal Prompt prefix dimension must match the raw FastWAM context dimension"
            )
        # ``behavior_indexes`` are understanding tokens in Zeva even when the
        # PIM list is empty.  Empty PIM disables only the residual; the task
        # base prefix remains a valid behavior condition.
        slot_mask = torch.ones(
            (context.shape[0], 1), device=context.device, dtype=torch.bool
        )
        return (
            torch.cat(
                (prefix.to(device=context.device, dtype=context.dtype), context),
                dim=1,
            ),
            torch.cat(
                (slot_mask, context_mask.to(device=context.device, dtype=torch.bool)),
                dim=1,
            ),
        )
