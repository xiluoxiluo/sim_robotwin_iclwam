"""Causal brief and persistent interaction memories."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class BriefInteractionTrace:
    effects: Tensor
    valid: Tensor
    source_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.effects.ndim != 3 or self.valid.shape != self.effects.shape[:2]:
            raise ValueError("BIT effects must be [B,L,D] and valid must be [B,L]")
        if self.source_indices and len(self.source_indices) != self.effects.shape[1]:
            raise ValueError("BIT source_indices must match its sequence length")


@dataclass(frozen=True)
class PersistentInteractionMemoryConfig:
    phase_dim: int = 128
    effect_dim: int = 128
    capacity: int = 256
    top_k: int = 4
    merge_threshold: float = 0.85
    phase_merge_weight: float = 0.5
    effect_merge_weight: float = 0.5

    def __post_init__(self) -> None:
        if min(self.phase_dim, self.effect_dim, self.capacity, self.top_k) < 1 or self.top_k > self.capacity:
            raise ValueError("Invalid PIM dimensions/capacity/top_k")
        if not -1 <= self.merge_threshold <= 1:
            raise ValueError("merge_threshold must be in [-1,1]")
        if (
            not math.isfinite(float(self.merge_threshold))
            or not math.isfinite(float(self.phase_merge_weight))
            or not math.isfinite(float(self.effect_merge_weight))
        ):
            raise ValueError("PIM thresholds and merge weights must be finite")
        if self.phase_merge_weight < 0 or self.effect_merge_weight < 0:
            raise ValueError("PIM merge weights must be non-negative")
        if self.phase_merge_weight + self.effect_merge_weight <= 0:
            raise ValueError("At least one PIM merge weight must be positive")


@dataclass
class PIMMemoryEntry:
    task_cluster: str
    phase: Tensor
    effect: Tensor
    attempt_id: int = 0
    transition_index: int = 0
    observation_count: int = 1
    metadata: dict[str, object] = field(default_factory=dict)
    # Keep the official Zeva first/last-attempt provenance while retaining the
    # backwards-compatible ``attempt_id`` field used by the RoboTwin adapter.
    first_attempt_id: int | None = None
    last_attempt_id: int | None = None

    def __post_init__(self) -> None:
        if self.first_attempt_id is None:
            self.first_attempt_id = int(self.attempt_id)
        if self.last_attempt_id is None:
            self.last_attempt_id = int(self.attempt_id)

    @property
    def merge_count(self) -> int:
        """Official-Zeva naming alias for the running observation count."""
        return int(self.observation_count)

    def cpu(self) -> "PIMMemoryEntry":
        return PIMMemoryEntry(
            self.task_cluster,
            self.phase.detach().float().cpu(),
            self.effect.detach().float().cpu(),
            self.attempt_id,
            self.transition_index,
            self.observation_count,
            dict(self.metadata),
            self.first_attempt_id,
            self.last_attempt_id,
        )


class PersistentInteractionMemory:
    """Episode-scoped non-parametric store retained across attempts."""

    def __init__(self, config: PersistentInteractionMemoryConfig | None = None) -> None:
        self.config = config or PersistentInteractionMemoryConfig()
        self._task_cluster: str | None = None
        self._episode_id: str | None = None
        self._attempt_id: int | None = None
        self._entries: list[PIMMemoryEntry] = []

    @property
    def entries(self) -> tuple[PIMMemoryEntry, ...]:
        return tuple(self._entries)

    @property
    def task_cluster(self) -> str | None:
        return self._task_cluster

    @property
    def attempt_id(self) -> int | None:
        return self._attempt_id

    def __len__(self) -> int:
        return len(self._entries)

    def reset_episode(self, task_cluster: str, *, episode_id: str | None = None) -> None:
        if not task_cluster:
            raise ValueError("task_cluster cannot be empty")
        self._task_cluster, self._episode_id, self._attempt_id = str(task_cluster), episode_id, 0
        self._entries.clear()

    def begin_attempt(self, attempt_id: int) -> None:
        if self._task_cluster is None or self._attempt_id is None:
            raise RuntimeError("reset_episode() is required before begin_attempt()")
        attempt_id = int(attempt_id)
        if attempt_id not in {self._attempt_id, self._attempt_id + 1}:
            raise ValueError("attempt_id must stay constant or increment by one")
        self._attempt_id = attempt_id

    def append_completed(self, *, task_cluster: str, phase: Tensor, effect: Tensor, attempt_id: int, transition_index: int = 0, metadata: dict[str, object] | None = None) -> tuple[int, bool]:
        if self._task_cluster is None:
            raise RuntimeError("reset_episode() is required before PIM writes")
        if str(task_cluster) != self._task_cluster:
            raise ValueError("PIM task mismatch")
        if phase.shape != (self.config.phase_dim,) or effect.shape != (self.config.effect_dim,):
            raise ValueError("PIM phase/effect shape mismatch")
        if not torch.isfinite(phase).all() or not torch.isfinite(effect).all():
            raise ValueError("PIM refuses non-finite values")
        phase = F.normalize(phase.detach().float().cpu(), dim=0)
        effect = F.normalize(effect.detach().float().cpu(), dim=0)
        # Advance the attempt only after the feature payload has passed all
        # validation, so a malformed write cannot mutate lifecycle state.
        self.begin_attempt(attempt_id)
        best_index, best_score = -1, -float("inf")
        denom = self.config.phase_merge_weight + self.config.effect_merge_weight
        for index, entry in enumerate(self._entries):
            score = self.config.phase_merge_weight * float(entry.phase @ phase) + self.config.effect_merge_weight * float(entry.effect @ effect)
            score /= denom
            if score > best_score:
                best_index, best_score = index, score
        if best_index >= 0 and best_score >= self.config.merge_threshold:
            entry = self._entries[best_index]
            count = entry.observation_count
            entry.phase = F.normalize((entry.phase * count + phase) / (count + 1), dim=0)
            entry.effect = F.normalize((entry.effect * count + effect) / (count + 1), dim=0)
            entry.observation_count = count + 1
            entry.attempt_id = int(attempt_id)
            entry.last_attempt_id = int(attempt_id)
            entry.transition_index = int(transition_index)
            entry.metadata.update(dict(metadata or {}))
            entry.metadata["last_merge_score"] = best_score
            return best_index, True
        if len(self._entries) >= self.config.capacity:
            evict = min(range(len(self._entries)), key=lambda i: (self._entries[i].observation_count, self._entries[i].attempt_id, self._entries[i].transition_index, i))
            self._entries.pop(evict)
        self._entries.append(
            PIMMemoryEntry(
                str(task_cluster),
                phase,
                effect,
                int(attempt_id),
                int(transition_index),
                1,
                dict(metadata or {}),
                int(attempt_id),
                int(attempt_id),
            )
        )
        return len(self._entries) - 1, False

    def query_tensors(self, phase: Tensor, *, top_k: int | None = None, exclude_attempt_id: int | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor, list[dict[str, object]]]:
        k = self.config.top_k if top_k is None else int(top_k)
        if k < 1 or k > self.config.capacity:
            raise ValueError("top_k outside configured capacity")
        if phase.shape != (self.config.phase_dim,):
            raise ValueError("PIM query phase shape mismatch")
        if not torch.isfinite(phase).all():
            raise ValueError("PIM refuses a non-finite query phase")
        candidates = [(i, e) for i, e in enumerate(self._entries) if exclude_attempt_id is None or e.attempt_id != exclude_attempt_id]
        if candidates:
            query = F.normalize(phase.detach().float().cpu(), dim=0)
            scores_all = torch.tensor([float(F.normalize(e.phase, dim=0) @ query) for _, e in candidates], dtype=torch.float32)
            order = torch.argsort(scores_all, descending=True, stable=True)[:k]
            selected = [candidates[int(i)] for i in order]
            scores = scores_all[order]
        else:
            selected, scores = [], torch.empty(0)
        phases = torch.zeros((k, self.config.phase_dim), dtype=torch.float32)
        effects = torch.zeros((k, self.config.effect_dim), dtype=torch.float32)
        valid = torch.zeros(k, dtype=torch.bool)
        metadata: list[dict[str, object]] = []
        for j, ((_, entry), score) in enumerate(zip(selected, scores, strict=True)):
            phases[j], effects[j], valid[j] = entry.phase, entry.effect, True
            source = dict(entry.metadata)
            source.update({
                "episode_id": self._episode_id,
                "task_cluster": entry.task_cluster,
                "attempt_id": entry.attempt_id,
                "first_attempt_id": entry.first_attempt_id,
                "last_attempt_id": entry.last_attempt_id,
                "transition_index": entry.transition_index,
                "observation_count": entry.observation_count,
                "score": float(score),
            })
            metadata.append(source)
        return phases, effects, valid, torch.cat((scores, torch.full((k - len(scores),), -torch.inf))), metadata


class BriefInteractionTraceBuffer:
    def __init__(self, effect_dim: int, max_length: int = 4) -> None:
        if effect_dim < 1 or max_length < 1:
            raise ValueError("effect_dim and max_length must be positive")
        self.effect_dim, self.max_length = effect_dim, max_length
        self._effects: list[Tensor] = []
        self._indices: list[int] = []

    def reset(self) -> None:
        self._effects.clear(); self._indices.clear()

    def append(self, effect: Tensor, transition_index: int) -> None:
        if effect.shape != (self.effect_dim,):
            raise ValueError("effect shape mismatch")
        self._effects.append(effect.detach().float().cpu()); self._indices.append(int(transition_index))
        del self._effects[:-self.max_length]; del self._indices[:-self.max_length]

    def tensors(self, device: torch.device | str | None = None) -> BriefInteractionTrace:
        effects = torch.zeros((1, self.max_length, self.effect_dim), dtype=torch.float32, device=device)
        valid = torch.zeros((1, self.max_length), dtype=torch.bool, device=device)
        if self._effects:
            values = torch.stack(self._effects).to(device=device)
            effects[0, -len(values):], valid[0, -len(values):] = values, True
        return BriefInteractionTrace(effects, valid, tuple([-1] * (self.max_length - len(self._indices)) + self._indices))
