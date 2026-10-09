"""Online Zeva memory lifecycle and full-history CTE runner."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import torch
from torch import Tensor

from .causal_transition_encoder import CausalTransitionEncoder
from .memory import BriefInteractionTraceBuffer, PersistentInteractionMemory


@dataclass(frozen=True)
class LifecycleConfig:
    bit_size: int = 4
    transition_steps: int = 4
    effect_window_transitions: int = 4
    action_dim: int = 14

    def __post_init__(self) -> None:
        if min(self.bit_size, self.transition_steps, self.effect_window_transitions, self.action_dim) < 1:
            raise ValueError("lifecycle dimensions must be positive")


class CausalCTEHistory:
    """Cache boundaries/actions and call CTE through its full-history API."""

    def __init__(
        self,
        cte: CausalTransitionEncoder,
        frame_encoder: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        self.cte = cte
        # ``None`` means that the caller already supplies the CTE tensor.  A
        # callable is deliberately kept outside CTE so the same history
        # runner can use Zeva's frozen Wan-VAE boundary when required.
        self.frame_encoder = frame_encoder
        self.reset()

    def reset(self, initial_frame: Tensor | None = None, *, initial_raw_step: int = 0) -> None:
        self._frames: list[Tensor] = []
        self._encoded_frames: list[Tensor] = []
        self._raw_steps: list[int] = []
        self._actions: list[Tensor] = []
        self._action_valid: list[Tensor] = []
        if initial_frame is not None:
            if initial_frame.ndim == 3:
                initial_frame = initial_frame.unsqueeze(0)
            if initial_frame.ndim != 4 or initial_frame.shape[0] != 1:
                raise ValueError("initial_frame must be [C,H,W] or [1,C,H,W]")
            frame = initial_frame[0].detach()
            initial_raw_step = int(initial_raw_step)
            if initial_raw_step < 0:
                raise ValueError("initial_raw_step must be non-negative")
            encoded = self._encode_boundary(frame) if self.frame_encoder is not None else None
            self._raw_steps.append(initial_raw_step)
            if encoded is not None:
                self._encoded_frames.append(encoded)
            else:
                self._frames.append(frame)

    def _encode_boundary(self, frame: Tensor) -> Tensor:
        """Encode one newly observed RGB boundary and keep only its latent."""
        if self.frame_encoder is None:
            raise RuntimeError("_encode_boundary requires a frame_encoder")
        with torch.no_grad():
            encoded = self.frame_encoder(frame.unsqueeze(0).unsqueeze(0))
        if encoded.ndim != 5 or encoded.shape[:2] != (1, 1):
            raise ValueError("frame_encoder must return [1,1,C,H,W] for one boundary")
        if encoded.shape[2] != self.cte.cfg.image_channels:
            raise ValueError(
                f"encoded frame channels {encoded.shape[2]} do not match CTE image_channels "
                f"{self.cte.cfg.image_channels}"
            )
        # Latents are frozen features; keeping them on CPU avoids retaining a
        # growing GPU tensor list during long online episodes. They are moved
        # back to the CTE device in forward().
        return encoded[0, 0].detach().float().cpu()

    def append_transition(
        self,
        action_group: Tensor,
        after_frame: Tensor,
        action_valid: Tensor | None = None,
        raw_step: int | None = None,
    ) -> None:
        if action_group.ndim == 3 and action_group.shape[0] == 1:
            action_group = action_group[0]
        if action_group.shape != (self.cte.cfg.transition_steps, self.cte.cfg.action_dim):
            raise ValueError("action_group shape does not match CTE config")
        if after_frame.ndim == 4 and after_frame.shape[0] == 1:
            after_frame = after_frame[0]
        if after_frame.ndim != 3:
            raise ValueError("after_frame must be [C,H,W] or [1,C,H,W]")
        if not self._raw_steps:
            raise RuntimeError("reset(initial_frame=...) is required before append_transition")
        if action_valid is None:
            action_valid = torch.ones(self.cte.cfg.transition_steps, dtype=torch.bool)
        action_valid = action_valid.flatten().bool()
        if action_valid.shape != (self.cte.cfg.transition_steps,):
            raise ValueError("action_valid must have one flag per raw action")
        expected_raw_step = self._raw_steps[-1] + self.cte.cfg.transition_steps
        if raw_step is None:
            raw_step = expected_raw_step
        raw_step = int(raw_step)
        if raw_step != expected_raw_step:
            raise ValueError(
                f"raw_step must advance by transition_steps={self.cte.cfg.transition_steps}; "
                f"expected {expected_raw_step}, got {raw_step}"
            )
        frame = after_frame.detach()
        encoded = self._encode_boundary(frame) if self.frame_encoder is not None else None
        self._actions.append(action_group.detach())
        self._action_valid.append(action_valid.detach())
        self._raw_steps.append(raw_step)
        if encoded is not None:
            self._encoded_frames.append(encoded)
        else:
            self._frames.append(frame)

    @torch.no_grad()
    def forward(self) -> dict[str, Tensor]:
        if not self._raw_steps:
            raise RuntimeError("CTE history is empty; call reset(initial_frame=...) first")
        cte_parameter = next(self.cte.parameters())
        device, dtype = cte_parameter.device, cte_parameter.dtype
        if self.frame_encoder is not None:
            if len(self._encoded_frames) != len(self._raw_steps):
                raise RuntimeError("encoded CTE history is out of sync with raw boundaries")
            frames = torch.stack(self._encoded_frames).unsqueeze(0).to(device=device)
        else:
            if len(self._frames) != len(self._raw_steps):
                raise RuntimeError("RGB CTE history is out of sync with raw boundaries")
            frames = torch.stack(self._frames).unsqueeze(0).to(device=device)
        # Direct RGB debug checkpoints and external adapters may produce a
        # dtype different from the CTE weights (e.g. FastWAM bf16 vs. CTE
        # float32).  Normalize at this boundary, never inside CTE itself.
        frames = frames.to(device=device, dtype=dtype)
        if frames.ndim != 5 or frames.shape[0] != 1 or frames.shape[1] != len(self._raw_steps):
            raise ValueError("frame_encoder must return [B,T,C,H,W] with the same batch/time axes")
        if frames.shape[2] != self.cte.cfg.image_channels:
            raise ValueError(
                f"encoded frame channels {frames.shape[2]} do not match CTE image_channels "
                f"{self.cte.cfg.image_channels}"
            )
        if self._actions:
            actions = torch.stack(self._actions).unsqueeze(0).to(device=device, dtype=dtype)
        else:
            actions = torch.zeros(
                (1, 0, self.cte.cfg.transition_steps, self.cte.cfg.action_dim),
                device=device,
                dtype=frames.dtype,
            )
        valid = torch.ones((1, frames.shape[1]), dtype=torch.bool, device=device)
        transition_valid = (
            torch.stack(self._action_valid).unsqueeze(0).to(device=device)
            if self._action_valid
            else torch.zeros((1, 0, self.cte.cfg.transition_steps), dtype=torch.bool, device=device)
        )
        return self.cte(frames, actions, valid_mask=valid, transition_valid=transition_valid)

    def current_phase(self) -> Tensor:
        return self.forward()["phase"][:, -1]

    def phase_at_raw_step(self, raw_step: int) -> Tensor:
        """Return the phase query at an observed raw-action boundary."""
        raw_step = int(raw_step)
        try:
            index = self._raw_steps.index(raw_step)
        except ValueError as exc:
            raise KeyError(f"raw_step {raw_step} is not an observed CTE boundary") from exc
        return self.forward()["phase"][:, index]


class CausalMemoryLifecycle:
    """Keep attempt-scoped BIT and episode-scoped PIM memory separate."""

    def __init__(self, pim: PersistentInteractionMemory, config: LifecycleConfig | None = None) -> None:
        self.config = config or LifecycleConfig()
        self.pim = pim
        self._effect_dim = int(pim.config.effect_dim)
        self.bit = BriefInteractionTraceBuffer(self._effect_dim, self.config.bit_size)
        self._attempt_id = 0
        self._effect_index = 0
        self.last_retrieval_sources: list[dict[str, object]] = []
        self.last_retrieval: dict[str, object] | None = None

    @property
    def effect_index(self) -> int:
        return self._effect_index

    @property
    def _transition_index(self) -> int:
        """Backward-compatible name; the counter now advances per effect."""
        return self._effect_index

    def reset_episode(self, task_cluster: str, episode_id: str | None = None) -> None:
        self.pim.reset_episode(task_cluster, episode_id=episode_id)
        self.bit.reset()
        self._attempt_id = 0
        self._effect_index = 0
        self.last_retrieval_sources = []
        self.last_retrieval = None

    def end_attempt(self) -> None:
        if self.pim.task_cluster is None:
            raise RuntimeError("reset_episode() is required before end_attempt()")
        # BIT is scoped to the just-finished attempt.  Clear it immediately,
        # including when there will be no retry, so stale short-term evidence
        # cannot be observed after the lifecycle boundary.
        self.bit.reset()

    def reset_attempt(self, attempt_id: int) -> None:
        attempt_id = int(attempt_id)
        if attempt_id not in {self._attempt_id, self._attempt_id + 1}:
            raise ValueError("attempt_id must increment by one")
        if attempt_id == self._attempt_id + 1:
            self.end_attempt()
            self.pim.begin_attempt(attempt_id)
        self._attempt_id = attempt_id
        self._effect_index = 0
        self.bit.reset()
        self.last_retrieval_sources = []
        self.last_retrieval = None

    def set_effect_dim(self, effect_dim: int) -> None:
        if effect_dim < 1:
            raise ValueError("effect_dim must be positive")
        if effect_dim != self._effect_dim:
            self._effect_dim = int(effect_dim)
            self.bit = BriefInteractionTraceBuffer(self._effect_dim, self.config.bit_size)

    def observe_completed_effect(
        self,
        phase_at_window_start: Tensor,
        effect_post: Tensor,
        *,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if self.pim.task_cluster is None:
            raise RuntimeError("reset_episode() is required before observing effects")
        if phase_at_window_start.ndim == 2 and phase_at_window_start.shape[0] == 1:
            phase_at_window_start = phase_at_window_start[0]
        if effect_post.ndim == 2 and effect_post.shape[0] == 1:
            effect_post = effect_post[0]
        if phase_at_window_start.ndim != 1 or effect_post.ndim != 1:
            raise ValueError("online phase/effect must be vectors")
        if phase_at_window_start.shape != (self.pim.config.phase_dim,) or effect_post.shape != (self._effect_dim,):
            raise ValueError("online phase/effect shape mismatch")
        effect_index = self._effect_index
        self.bit.append(effect_post, effect_index)
        # A completed effect is immediately persistent. This is the causal
        # boundary: later queries in the same attempt may retrieve it, while
        # incomplete/future effects are never passed to this method.
        self.pim.append_completed(
            task_cluster=self.pim.task_cluster,
            phase=phase_at_window_start.detach(),
            effect=effect_post.detach(),
            attempt_id=self._attempt_id,
            transition_index=effect_index * self.config.effect_window_transitions,
            metadata={**dict(metadata or {}), "effect_index": effect_index},
        )
        self._effect_index += 1

    def observe_completed_transition(
        self, phase: Tensor, effect: Tensor, *, metadata: dict[str, object] | None = None
    ) -> None:
        """Compatibility alias for older callers; cadence is effect-level."""
        self.observe_completed_effect(phase, effect, metadata=metadata)

    def memory_inputs(self, phase: Tensor, task_tokens: Tensor) -> dict[str, Tensor]:
        bit = self.bit.tensors(device=phase.device)
        phase_query = phase[0] if phase.ndim == 2 and phase.shape[0] == 1 else phase
        if phase_query.ndim != 1:
            raise ValueError("online phase must be [D] or [1,D]")
        phases, effects, valid, scores, sources = self.pim.query_tensors(
            phase_query.detach().float().cpu(),
            top_k=self.pim.config.top_k,
            # Completed interactions are causally visible immediately,
            # including those produced earlier in the current attempt.
            exclude_attempt_id=None,
        )
        self.last_retrieval_sources = sources
        self.last_retrieval = {
            "query_phase": phase_query.detach().float().cpu(),
            "phases": phases,
            "effects": effects,
            "valid": valid,
            "scores": scores,
            "sources": sources,
        }
        return {
            "task_tokens": task_tokens,
            "current_phase": phase_query.unsqueeze(0),
            "bit_effects": bit.effects.to(phase.device),
            "bit_mask": bit.valid.to(phase.device),
            "pim_phases": phases.unsqueeze(0).to(phase.device),
            "pim_effects": effects.unsqueeze(0).to(phase.device),
            "pim_mask": valid.unsqueeze(0).to(phase.device),
        }
