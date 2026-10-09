"""Trainable task/phase/BIT/PIM fusion for FastWAM."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def task_tokens_from_context(
    context: Tensor, context_mask: Tensor | None, output_dim: int
) -> Tensor:
    """Compress full text context into the prompt encoder global token."""
    if context.ndim != 3 or output_dim < 1:
        raise ValueError("context must be [B,L,D] and output_dim must be positive")
    if context_mask is None:
        pooled = context.float().mean(dim=1)
    else:
        if context_mask.shape != context.shape[:2]:
            raise ValueError("context_mask must be [B,L]")
        weights = context_mask.to(device=context.device, dtype=torch.float32).unsqueeze(-1)
        pooled = (context.float() * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    pooled = F.adaptive_avg_pool1d(pooled.unsqueeze(1), output_dim).squeeze(1)
    return pooled.to(dtype=context.dtype)


@dataclass(frozen=True)
class CausalPromptConfig:
    global_dim: int = 256
    phase_dim: int = 128
    effect_dim: int = 128
    brief_length: int = 4
    persistent_length: int = 4
    hidden_dim: int = 256
    num_heads: int = 4
    # Stage-2 regularization: support=BIT slots, context=PIM slots.
    # Applied only while this encoder is in training mode; inference keeps
    # the complete retrieved memory unchanged.
    pim_context_dropout: float = 0.0
    pim_support_dropout: float = 0.0

    def __post_init__(self) -> None:
        if min(
            self.global_dim,
            self.phase_dim,
            self.effect_dim,
            self.brief_length,
            self.persistent_length,
            self.hidden_dim,
            self.num_heads,
        ) < 1:
            raise ValueError("prompt dimensions, lengths, and heads must be positive")
        if self.hidden_dim % self.num_heads:
            raise ValueError("prompt hidden_dim must be divisible by num_heads")
        if not 0 <= self.pim_context_dropout < 1 or not 0 <= self.pim_support_dropout < 1:
            raise ValueError("PIM context/support dropout must be in [0,1)")


class CausalPromptEncoder(nn.Module):
    def __init__(self, config: CausalPromptConfig | None = None) -> None:
        super().__init__(); self.config = config or CausalPromptConfig(); cfg = self.config
        self.global_project = nn.Sequential(nn.LayerNorm(cfg.global_dim), nn.Linear(cfg.global_dim, cfg.hidden_dim))
        self.phase_project = nn.Sequential(nn.LayerNorm(cfg.phase_dim), nn.Linear(cfg.phase_dim, cfg.hidden_dim))
        self.effect_project = nn.Sequential(nn.LayerNorm(cfg.effect_dim), nn.Linear(cfg.effect_dim, cfg.hidden_dim))
        self.brief_position = nn.Parameter(torch.empty(1, cfg.brief_length, cfg.hidden_dim))
        self.persistent_position = nn.Parameter(torch.empty(1, cfg.persistent_length, cfg.hidden_dim))
        self.bos_brief = nn.Parameter(torch.empty(1, 1, cfg.hidden_dim))
        self.bos_persistent = nn.Parameter(torch.empty(1, 1, cfg.hidden_dim))
        self.brief_attention = nn.MultiheadAttention(cfg.hidden_dim, cfg.num_heads, batch_first=True)
        self.persistent_attention = nn.MultiheadAttention(cfg.hidden_dim, cfg.num_heads, batch_first=True)
        self.fusion = nn.Sequential(nn.LayerNorm(3 * cfg.hidden_dim), nn.Linear(3 * cfg.hidden_dim, cfg.hidden_dim), nn.SiLU(), nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        self.brief_attention._reset_parameters()
        self.persistent_attention._reset_parameters()
        nn.init.normal_(self.brief_position, std=0.02)
        nn.init.normal_(self.persistent_position, std=0.02)
        nn.init.normal_(self.bos_brief, std=0.02)
        nn.init.normal_(self.bos_persistent, std=0.02)

    def forward(self, task_tokens: Tensor, current_phase: Tensor, bit_effects: Tensor, bit_mask: Tensor, pim_phases: Tensor, pim_effects: Tensor, pim_mask: Tensor) -> Tensor:
        cfg = self.config; batch = task_tokens.shape[0]
        if task_tokens.shape != (batch, cfg.global_dim) or current_phase.shape != (batch, cfg.phase_dim) or bit_effects.shape != (batch, cfg.brief_length, cfg.effect_dim) or bit_mask.shape != (batch, cfg.brief_length) or pim_phases.shape != (batch, cfg.persistent_length, cfg.phase_dim) or pim_effects.shape != (batch, cfg.persistent_length, cfg.effect_dim) or pim_mask.shape != (batch, cfg.persistent_length):
            raise ValueError("Causal prompt input shapes do not match config")
        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        task_tokens = task_tokens.to(device=device)
        current_phase = current_phase.to(device=device)
        bit_effects = bit_effects.to(device=device)
        bit_mask = bit_mask.to(device=device)
        pim_phases = pim_phases.to(device=device)
        pim_effects = pim_effects.to(device=device)
        pim_mask = pim_mask.to(device=device)
        task_tokens = task_tokens.to(dtype=dtype)
        current_phase = current_phase.to(dtype=dtype)
        bit_effects = bit_effects.to(dtype=dtype)
        pim_phases = pim_phases.to(dtype=dtype)
        pim_effects = pim_effects.to(dtype=dtype)
        query = self.global_project(task_tokens) + self.phase_project(current_phase)
        bit = self.effect_project(bit_effects) + self.brief_position
        bit_mask = bit_mask.bool()
        if self.training and cfg.pim_support_dropout:
            keep = torch.rand(bit_mask.shape, device=device) >= cfg.pim_support_dropout
            bit_mask = bit_mask & keep
        bit = torch.where(bit_mask.unsqueeze(-1), bit, self.bos_brief.expand(batch, -1, -1))
        bit_padding = ~bit_mask; empty = ~bit_mask.any(dim=-1); bit_padding = bit_padding.clone(); bit_padding[empty, 0] = False
        bit_context, _ = self.brief_attention(query[:, None], bit, bit, key_padding_mask=bit_padding, need_weights=False)
        pim = self.phase_project(pim_phases) + self.effect_project(pim_effects) + self.persistent_position
        pim_mask = pim_mask.bool()
        if self.training and cfg.pim_context_dropout:
            keep = torch.rand(pim_mask.shape, device=device) >= cfg.pim_context_dropout
            pim_mask = pim_mask & keep
        pim = torch.where(pim_mask.unsqueeze(-1), pim, self.bos_persistent.expand(batch, -1, -1))
        pim_padding = ~pim_mask; empty = ~pim_mask.any(dim=-1); pim_padding = pim_padding.clone(); pim_padding[empty, 0] = False
        pim_context, _ = self.persistent_attention(query[:, None], pim, pim, key_padding_mask=pim_padding, need_weights=False)
        causal_prompt = self.fusion(
            torch.cat((query, bit_context[:, 0], pim_context[:, 0]), dim=-1)
        )
        return causal_prompt
