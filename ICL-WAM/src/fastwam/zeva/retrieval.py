"""Deterministic phase/effect-conditioned offline retrieval.

The normal Zeva PIM path is phase-conditioned and same-task.  Cross-task
effect transfer is deliberately exposed as a separate API so an experiment
cannot silently change the semantics of the default PIM proxy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class RetrievalResult:
    phases: Tensor
    effects: Tensor
    mask: Tensor
    scores: Tensor
    sources: tuple[dict[str, object], ...]


class MemoryBank:
    """Offline proxy bank for phase retrieval and cross-task effect transfer."""

    def __init__(self, phase_dim: int = 128, effect_dim: int = 128, top_k: int = 4, same_task_only: bool = True) -> None:
        self.phase_dim, self.effect_dim, self.top_k, self.same_task_only = int(phase_dim), int(effect_dim), int(top_k), bool(same_task_only)
        self._rows: list[dict[str, object]] = []

    def add(
        self,
        phase: Tensor,
        effect: Tensor,
        *,
        episode_id: str,
        task_id: int | str,
        attempt_id: int = 0,
        transition_index: int = 0,
        effect_index: int | None = None,
        episode_step: int | None = None,
        window_index: int | None = None,
        valid: bool = True,
    ) -> None:
        if valid:
            if phase.shape != (self.phase_dim,) or effect.shape != (self.effect_dim,):
                raise ValueError("memory bank feature shape mismatch")
            if not torch.isfinite(phase).all() or not torch.isfinite(effect).all():
                raise ValueError("memory bank refuses non-finite features")
            self._rows.append({"phase": F.normalize(phase.detach().float().cpu(), dim=0), "effect": F.normalize(effect.detach().float().cpu(), dim=0), "episode_id": str(episode_id), "task_id": task_id, "attempt_id": int(attempt_id), "transition_index": int(transition_index), "effect_index": None if effect_index is None else int(effect_index), "episode_step": None if episode_step is None else int(episode_step), "window_index": None if window_index is None else int(window_index)})

    def _retrieve(
        self,
        query: Tensor,
        *,
        key: str,
        query_name: str,
        episode_id: str | None,
        task_id: int | str | None,
        top_k: int | None,
        same_task_only: bool,
        require_cross_task: bool = False,
    ) -> RetrievalResult:
        k = self.top_k if top_k is None else int(top_k)
        if k < 1:
            raise ValueError("top_k must be positive")
        expected_dim = self.phase_dim if key == "phase" else self.effect_dim
        if query.shape != (expected_dim,):
            raise ValueError(f"{query_name} shape mismatch")
        if not torch.isfinite(query).all():
            raise ValueError(f"memory bank refuses a non-finite {query_name}")
        if require_cross_task and task_id is None:
            raise ValueError("cross-task effect retrieval requires task_id")
        query = F.normalize(query.detach().float().cpu(), dim=0)
        rows = []
        for row in self._rows:
            if episode_id is not None and row["episode_id"] == str(episode_id):
                continue
            if require_cross_task:
                if row["task_id"] == task_id:
                    continue
            elif same_task_only and task_id is not None and row["task_id"] != task_id:
                continue
            rows.append(row)
        scores = torch.tensor([float(row[key] @ query) for row in rows], dtype=torch.float32) if rows else torch.empty(0)
        if rows:
            order = torch.argsort(scores, descending=True, stable=True)[:k]
            rows, scores = [rows[int(i)] for i in order], scores[order]
        phases = torch.zeros((k, self.phase_dim), dtype=torch.float32)
        effects = torch.zeros((k, self.effect_dim), dtype=torch.float32)
        mask = torch.zeros(k, dtype=torch.bool)
        sources: list[dict[str, object]] = []
        for i, row in enumerate(rows[:k]):
            phases[i], effects[i], mask[i] = row["phase"], row["effect"], True
            source = {name: row[name] for name in ("episode_id", "task_id", "attempt_id", "transition_index", "effect_index", "episode_step", "window_index")}
            source["retrieval_key"] = key
            source["retrieval_scope"] = "cross_task" if require_cross_task else ("same_task" if same_task_only else "all_tasks")
            sources.append(source)
        padded_scores = torch.full((k,), -torch.inf, dtype=torch.float32); padded_scores[:len(scores)] = scores
        return RetrievalResult(phases, effects, mask, padded_scores, tuple(sources))

    def retrieve(
        self,
        query_phase: Tensor,
        *,
        episode_id: str | None = None,
        task_id: int | str | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Retrieve by phase using the default same-task PIM proxy policy."""
        return self._retrieve(
            query_phase,
            key="phase",
            query_name="query_phase",
            episode_id=episode_id,
            task_id=task_id,
            top_k=top_k,
            same_task_only=self.same_task_only,
        )

    def retrieve_by_effect(
        self,
        query_effect: Tensor,
        *,
        episode_id: str | None = None,
        task_id: int | str | None = None,
        top_k: int | None = None,
        cross_task: bool = True,
    ) -> RetrievalResult:
        """Retrieve functionally similar effects from other tasks.

        This matches Zeva's separate cross-task effect-transfer analysis:
        ranking uses effect-token cosine similarity, the current episode is
        excluded, and ``cross_task=True`` excludes the query task itself.
        ``cross_task=False`` is available for controlled same-task ablations.
        """
        return self._retrieve(
            query_effect,
            key="effect",
            query_name="query_effect",
            episode_id=episode_id,
            task_id=task_id,
            top_k=top_k,
            same_task_only=self.same_task_only,
            require_cross_task=cross_task,
        )
