# SPDX-FileCopyrightText: Copyright (c) 2026 Z-Trans CORPORATION. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Adapted from air-embodied-brain/Zeva's
# ``cosmos_framework/model/zeva/causal_transition_encoder.py``.
#
# ICL-WAM adaptations are limited to RoboTwin's action width and data/mask
# boundary; the architecture and effect-window semantics remain canonical.

"""Causal Transition Encoder (CTE), copied from Zeva's canonical module.

ICL-WAM changes only the RoboTwin action width (14).  The CTE input, internal
heads, right-shifted action stream, and effect-window heads are kept identical
to Zeva's ``[B,T,C,H,W]`` tensor interface.  RGB/Wan-VAE conversion belongs to
the caller and is never performed in this module.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class CausalTransitionEncoderConfig:
    action_dim: int = 14
    hidden_dim: int = 256
    retrieval_dim: int = 128
    phase_dim: int = 128
    effect_dim: int = 128
    transition_steps: int = 4
    effect_window_transitions: int = 4
    effect_target_grid: tuple[int, int] = (4, 6)
    num_layers: int = 4
    num_heads: int = 8
    image_channels: int = 3
    ema_decay: float = 0.996
    use_mamba: bool = False
    vision_chunk_size: int = 256
    vision_gradient_checkpointing: bool = True
    vision_checkpoint_threshold: int = 1024

    def __post_init__(self) -> None:
        if min(self.action_dim, self.hidden_dim, self.retrieval_dim, self.phase_dim, self.effect_dim,
               self.transition_steps, self.effect_window_transitions, self.num_layers,
               self.num_heads, self.image_channels) < 1:
            raise ValueError("CTE dimensions and layer counts must be positive")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.phase_dim > self.hidden_dim:
            raise ValueError("phase_dim cannot exceed hidden_dim for visual_key projection")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class _VisionStem(nn.Module):
    """Small trainable visual stem for the CTE input representation."""

    def __init__(self, channels: int, hidden_dim: int, chunk_size: int, gradient_checkpointing: bool, checkpoint_threshold: int) -> None:
        super().__init__()
        width = max(hidden_dim // 4, 32)
        self.net = nn.Sequential(
            nn.Conv2d(channels, width, kernel_size=7, stride=4, padding=3), nn.GELU(),
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(width * 2, hidden_dim, kernel_size=3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.chunk_size = chunk_size
        self.gradient_checkpointing = gradient_checkpointing
        self.checkpoint_threshold = checkpoint_threshold

    def _encode_flat(self, flat: Tensor) -> Tensor:
        return self.proj(self.net(flat).flatten(1))

    def forward(self, frames: Tensor) -> Tensor:
        batch, steps = frames.shape[:2]
        flat = frames.flatten(0, 1)
        use_checkpoint = self.training and self.gradient_checkpointing and torch.is_grad_enabled() and flat.shape[0] > self.checkpoint_threshold
        if not use_checkpoint:
            return self._encode_flat(flat).unflatten(0, (batch, steps))
        chunks = [checkpoint(self._encode_flat, chunk, use_reentrant=False) for chunk in flat.split(self.chunk_size, dim=0)]
        return torch.cat(chunks, dim=0).unflatten(0, (batch, steps))


class _FrozenVAEDeltaTarget(nn.Module):
    """Fixed spatial projection used as Zeva's effect-delta target."""

    def __init__(self, channels: int, hidden_dim: int, grid: tuple[int, int]) -> None:
        super().__init__()
        input_dim = channels * grid[0] * grid[1]
        generator = torch.Generator().manual_seed(20260815)
        projection = torch.randn(hidden_dim, input_dim, generator=generator) / input_dim**0.5
        self.grid = grid
        self.register_buffer("projection", projection, persistent=True)

    @torch.no_grad()
    def forward(self, frames: Tensor) -> Tensor:
        batch, steps = frames.shape[:2]
        flat = frames.flatten(0, 1).float()
        pooled = F.adaptive_avg_pool2d(flat, self.grid).flatten(1)
        pooled = F.layer_norm(pooled, (pooled.shape[-1],))
        return F.linear(pooled, self.projection).unflatten(0, (batch, steps))


class _CausalMixer(nn.Module):
    def __init__(self, hidden_dim: int, use_mamba: bool) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.kind = "gru"
        if use_mamba:
            try:
                from mamba_ssm import Mamba  # type: ignore[import-not-found]
                self.mixer = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
                self.kind = "mamba"
            except ImportError:
                self.mixer = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        else:
            self.mixer = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(x)
        if self.kind == "mamba":
            y = self.mixer(y)
        else:
            y, _ = self.mixer(y)
        return x + y


class _CausalInteractionBlock(nn.Module):
    """Causal visual/action/interaction streams with same-step cross attention."""

    def __init__(self, cfg: CausalTransitionEncoderConfig) -> None:
        super().__init__()
        self.visual = _CausalMixer(cfg.hidden_dim, cfg.use_mamba)
        self.action = _CausalMixer(cfg.hidden_dim, cfg.use_mamba)
        self.interaction_state = _CausalMixer(cfg.hidden_dim, cfg.use_mamba)
        self.cross = nn.MultiheadAttention(cfg.hidden_dim, cfg.num_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(cfg.hidden_dim)
        self.ffn = nn.Sequential(nn.LayerNorm(cfg.hidden_dim), nn.Linear(cfg.hidden_dim, 4 * cfg.hidden_dim), nn.GELU(), nn.Linear(4 * cfg.hidden_dim, cfg.hidden_dim))

    def forward(
        self,
        visual: Tensor,
        action: Tensor,
        interaction_state: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        visual, action, interaction_state = (
            self.visual(visual),
            self.action(action),
            self.interaction_state(interaction_state),
        )
        tokens = torch.stack((visual, action, interaction_state), dim=2)
        batch, steps, streams, dim = tokens.shape
        tokens = tokens.reshape(batch * steps, streams, dim)
        attended, _ = self.cross(tokens, tokens, tokens, need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.ffn(tokens)
        tokens = tokens.reshape(batch, steps, streams, dim)
        visual, action, interaction_state = tokens.unbind(dim=2)
        mask = valid.unsqueeze(-1).to(visual.dtype)
        return visual * mask, action * mask, interaction_state * mask


class CausalTransitionEncoder(nn.Module):
    """Causal phase/effect encoder; action stream is right-shifted for leakage protection."""

    def __init__(self, cfg: CausalTransitionEncoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CausalTransitionEncoderConfig()
        cfg = self.cfg
        self.visual_encoder = _VisionStem(cfg.image_channels, cfg.hidden_dim, cfg.vision_chunk_size, cfg.vision_gradient_checkpointing, cfg.vision_checkpoint_threshold)
        self.target_visual_encoder = deepcopy(self.visual_encoder).requires_grad_(False)
        transition_dim = cfg.transition_steps * cfg.action_dim
        self.action_encoder = nn.Sequential(nn.LayerNorm(transition_dim), nn.Linear(transition_dim, cfg.hidden_dim))
        self.bos_action = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))
        self.interaction_state_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))
        nn.init.normal_(self.bos_action, std=0.02)
        nn.init.normal_(self.interaction_state_token, std=0.02)
        self.blocks = nn.ModuleList([_CausalInteractionBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = nn.LayerNorm(cfg.hidden_dim)
        self.retrieval_head = nn.Linear(cfg.hidden_dim, cfg.retrieval_dim)
        self.phase_head = nn.Linear(cfg.hidden_dim, cfg.phase_dim)
        self.action_head = nn.Sequential(nn.LayerNorm(cfg.hidden_dim), nn.Linear(cfg.hidden_dim, transition_dim))
        self.visual_head = nn.Sequential(nn.LayerNorm(cfg.hidden_dim), nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
        effect_window_dim = cfg.effect_window_transitions * transition_dim
        self.effect_action_encoder = nn.Sequential(nn.LayerNorm(effect_window_dim), nn.Linear(effect_window_dim, cfg.hidden_dim))
        self.frozen_effect_target = _FrozenVAEDeltaTarget(cfg.image_channels, cfg.hidden_dim, cfg.effect_target_grid)
        self.effect_pre_head = nn.Sequential(nn.LayerNorm(2 * cfg.hidden_dim), nn.Linear(2 * cfg.hidden_dim, cfg.effect_dim))
        self.effect_post_head = nn.Sequential(nn.LayerNorm(2 * cfg.hidden_dim), nn.Linear(2 * cfg.hidden_dim, cfg.effect_dim))
        self.effect_action_head = nn.Sequential(nn.LayerNorm(cfg.effect_dim), nn.Linear(cfg.effect_dim, 4 * cfg.action_dim))
        self.effect_outcome_head = nn.Sequential(nn.LayerNorm(cfg.effect_dim), nn.Linear(cfg.effect_dim, cfg.hidden_dim))

    @torch.no_grad()
    def update_ema_target(self) -> None:
        for target, online in zip(self.target_visual_encoder.parameters(), self.visual_encoder.parameters(), strict=True):
            target.lerp_(online, 1.0 - self.cfg.ema_decay)

    @torch.no_grad()
    def encode_target_vision(self, frames: Tensor) -> Tensor:
        return self.target_visual_encoder(frames)

    def forward(
        self,
        frames: Tensor,
        transition_actions: Tensor,
        valid_mask: Tensor | None = None,
        transition_valid: Tensor | None = None,
    ) -> dict[str, Tensor]:
        cfg = self.cfg
        if frames.ndim != 5 or transition_actions.ndim != 4:
            raise ValueError("Expected frames [B,T,C,H,W] and transition_actions [B,T-1,4,A].")
        if frames.shape[2] != cfg.image_channels:
            raise ValueError(
                f"frames channel count {frames.shape[2]} does not match "
                f"CTE image_channels={cfg.image_channels}"
            )
        if transition_actions.shape[0] != frames.shape[0] or transition_actions.shape[1] != frames.shape[1] - 1 or transition_actions.shape[2:] != (cfg.transition_steps, cfg.action_dim):
            raise ValueError("Frame/transition time axes or action shape do not match CTE config.")
        if valid_mask is None:
            valid_mask = torch.ones(frames.shape[:2], dtype=torch.bool, device=frames.device)
        if transition_valid is None:
            transition_valid = torch.ones(transition_actions.shape[:-1], dtype=torch.bool, device=frames.device)
        if valid_mask.shape != frames.shape[:2] or transition_valid.shape != transition_actions.shape[:-1]:
            raise ValueError("valid_mask must be [B,T] and transition_valid must be [B,T-1,4]")
        valid_mask = valid_mask.to(device=frames.device, dtype=torch.bool)
        transition_valid = transition_valid.to(device=frames.device, dtype=torch.bool)
        visual = self.visual_encoder(frames)
        transition_embed = self.action_encoder(transition_actions.flatten(-2))
        action = self.bos_action.expand(frames.shape[0], frames.shape[1], -1).clone()
        action[:, 1:] = transition_embed
        action[:, 1:] *= transition_valid.all(dim=-1, keepdim=True).to(action.dtype)
        interaction = self.interaction_state_token.expand_as(visual)
        for block in self.blocks:
            visual, action, interaction = block(visual, action, interaction, valid_mask)
        z = self.final_norm(interaction) * valid_mask.unsqueeze(-1).to(interaction.dtype)
        with torch.no_grad():
            target_visual = self.target_visual_encoder(frames)
        transition_complete = valid_mask[:, :-1] & valid_mask[:, 1:] & transition_valid.all(dim=-1)
        frozen_effect_visual = self.frozen_effect_target(frames)
        windows = (frames.shape[1] - 1) // cfg.effect_window_transitions
        if windows:
            starts = torch.arange(windows, device=frames.device) * cfg.effect_window_transitions
            ends = starts + cfg.effect_window_transitions
            effect_actions = transition_actions[:, : windows * cfg.effect_window_transitions].reshape(frames.shape[0], windows, cfg.effect_window_transitions, cfg.transition_steps, cfg.action_dim)
            effect_valid = transition_valid[:, : windows * cfg.effect_window_transitions].reshape(frames.shape[0], windows, cfg.effect_window_transitions, cfg.transition_steps)
            effect_complete = valid_mask[:, starts] & valid_mask[:, ends] & effect_valid.all(dim=(-1, -2))
            effect_embed = self.effect_action_encoder(effect_actions.flatten(-3))
            frozen_visual = frozen_effect_visual
            effect_delta = frozen_visual[:, ends] - frozen_visual[:, starts]
            effect_pre_raw = self.effect_pre_head(torch.cat((z[:, starts], effect_embed), dim=-1))
            effect_post_raw = self.effect_post_head(torch.cat((frozen_visual[:, starts], effect_delta), dim=-1))
        else:
            shape = (frames.shape[0], 0)
            effect_actions = transition_actions.new_zeros((frames.shape[0], 0, cfg.effect_window_transitions, cfg.transition_steps, cfg.action_dim))
            effect_complete = torch.zeros(shape, dtype=torch.bool, device=frames.device)
            effect_delta = z[:, :0]
            effect_pre_raw = z[:, :0]
            effect_post_raw = z[:, :0]
        if effect_pre_raw.shape[1] == 0:
            effect_pre = effect_post = effect_pre_raw.new_zeros((frames.shape[0], 0, cfg.effect_dim))
            effect_outcome_pre = effect_outcome_post = effect_pre_raw.new_zeros((frames.shape[0], 0, cfg.hidden_dim))
            effect_action = effect_pre_raw.new_zeros((frames.shape[0], 0, 4 * cfg.action_dim))
        else:
            effect_pre, effect_post = F.normalize(effect_pre_raw, dim=-1), F.normalize(effect_post_raw, dim=-1)
            effect_outcome_pre, effect_outcome_post = self.effect_outcome_head(effect_pre_raw), self.effect_outcome_head(effect_post_raw)
            effect_action = self.effect_action_head(effect_pre_raw)
        return {
            "causal_interaction_state": z,
            "retrieval": F.normalize(self.retrieval_head(z), dim=-1),
            "phase": F.normalize(self.phase_head(z), dim=-1),
            "next_action": self.action_head(z[:, :-1]).view(frames.shape[0], frames.shape[1] - 1, cfg.transition_steps, cfg.action_dim),
            "next_vision": self.visual_head(z),
            "visual_key": F.normalize(target_visual[..., : cfg.phase_dim], dim=-1),
            "effect_pre": effect_pre,
            "effect_post": effect_post,
            "effect_pre_raw": effect_pre_raw,
            "effect_post_raw": effect_post_raw,
            "effect_outcome_pre": effect_outcome_pre,
            "effect_outcome_post": effect_outcome_post,
            "effect_action": effect_action,
            "effect_actions": effect_actions,
            "effect_delta_target": effect_delta,
            "effect_complete": effect_complete,
            "target_visual": target_visual,
            "transition_complete": transition_complete,
        }
