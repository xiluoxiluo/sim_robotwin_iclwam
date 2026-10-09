"""FastWAM task-context prototype bank and retrieval interface.

The bank is deliberately separate from online BIT/PIM.  It stores static
task-level behavior prototypes, while BIT/PIM stores causal interaction
evidence collected during an episode.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor
from torch.nn import functional as F

from .causal_prompt import task_tokens_from_context


@dataclass(frozen=True)
class TaskContextRetrievalResult:
    """Values [B,V], indices/scores [B,K], and one source group per query.

    Direct rank-1 queries to ``retrieve`` omit B in the returned tensors.
    """

    values: Tensor
    indices: Tensor
    scores: Tensor
    sources: tuple[dict[str, Any], ...]


class TaskContextBank:
    """A frozen bank of FastWAM task behavior prototypes.

    Each entry has a ``retrieval_key`` in the FastWAM query space and a
    ``behavior_value`` consumed as the CausalPrompt global/task feature.  The
    two vectors are kept separate so a future FastWAM-specific retrieval head
    can be introduced without changing the artifact or call sites.
    """

    FORMAT = "fastwam_task_context_bank_v1"

    def __init__(self, entries: Iterable[dict[str, Any]], *, key_dim: int = 256, value_dim: int = 256, temperature: float = 0.07, metadata: dict[str, Any] | None = None) -> None:
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.temperature = float(temperature)
        self.metadata = dict(metadata or {})
        if self.key_dim < 1 or self.value_dim < 1:
            raise ValueError("task-context dimensions must be positive")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("task-context temperature must be finite and positive")
        self._entries: list[dict[str, Any]] = []
        for entry in entries:
            self._add_entry(entry)
        if not self._entries:
            raise ValueError("task-context bank must contain at least one entry")
        self._keys = torch.stack([entry["retrieval_key"] for entry in self._entries])
        self._values = torch.stack([entry["behavior_value"] for entry in self._entries])

    @property
    def entries(self) -> tuple[dict[str, Any], ...]:
        return tuple({**entry, "retrieval_key": entry["retrieval_key"].clone(), "behavior_value": entry["behavior_value"].clone()} for entry in self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def _add_entry(self, entry: dict[str, Any]) -> None:
        if not isinstance(entry, dict):
            raise ValueError("task-context bank entries must be dictionaries")
        if "retrieval_key" not in entry or "behavior_value" not in entry:
            raise ValueError("task-context bank entries require retrieval_key and behavior_value")
        key = torch.as_tensor(entry["retrieval_key"], dtype=torch.float32).detach().clone().cpu()
        value = torch.as_tensor(entry["behavior_value"], dtype=torch.float32).detach().clone().cpu()
        if key.shape != (self.key_dim,) or value.shape != (self.value_dim,):
            raise ValueError(
                "task-context bank entry dimensions do not match configuration: "
                f"key={tuple(key.shape)} value={tuple(value.shape)} expected=({self.key_dim},{self.value_dim})"
            )
        if not torch.isfinite(key).all() or not torch.isfinite(value).all():
            raise ValueError("task-context bank refuses non-finite entries")
        metadata = {k: v for k, v in entry.items() if k not in {"retrieval_key", "behavior_value"}}
        self._entries.append({
            "retrieval_key": F.normalize(key, dim=0),
            "behavior_value": value,
            **metadata,
        })

    @torch.no_grad()
    def retrieve(self, query: Tensor, *, top_k: int = 1, exclude_episode_ids: Iterable[str] | None = None) -> TaskContextRetrievalResult:
        """Retrieve a softmax-weighted task prototype for each query."""
        query = torch.as_tensor(query).float()
        single = query.ndim == 1
        if single:
            query = query.unsqueeze(0)
        if query.ndim != 2 or query.shape[-1] != self.key_dim:
            raise ValueError(f"query must be [B,{self.key_dim}] or [{self.key_dim}]")
        if not torch.isfinite(query).all():
            raise ValueError("task-context query contains non-finite values")
        if not 1 <= int(top_k) <= len(self._entries):
            raise ValueError(f"top_k must be in [1,{len(self._entries)}]")
        keys = self._keys.to(query.device)
        values = self._values.to(query.device)
        scores_all = F.normalize(query, dim=-1) @ keys.T
        if exclude_episode_ids is not None:
            episode_ids = tuple(str(value) for value in exclude_episode_ids)
            if len(episode_ids) != query.shape[0]:
                raise ValueError("exclude_episode_ids must contain one ID per query")
            eligible = torch.tensor([
                [str(entry.get("episode_id", "")) != episode_id for entry in self._entries]
                for episode_id in episode_ids
            ], device=query.device, dtype=torch.bool)
            if bool((eligible.sum(dim=-1) < int(top_k)).any()):
                raise ValueError("not enough bank entries after excluding the query episode")
            scores_all = scores_all.masked_fill(~eligible, -torch.inf)
        scores, indices = scores_all.topk(int(top_k), dim=-1, largest=True, sorted=True)
        weights = F.softmax(scores / self.temperature, dim=-1)
        selected = values[indices]
        output = torch.einsum("bk,bkd->bd", weights, selected)
        sources: list[dict[str, Any]] = []
        for rows in indices.tolist():
            sources.append({
                "top_k": [
                    {k: v for k, v in self._entries[row].items() if k not in {"retrieval_key", "behavior_value"}}
                    for row in rows
                ]
            })
        if single:
            output, indices, scores = output[0], indices[0], scores[0]
        return TaskContextRetrievalResult(output, indices, scores, tuple(sources))

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        payload = {
            "format": self.FORMAT,
            "config": {"key_dim": self.key_dim, "value_dim": self.value_dim, "temperature": self.temperature},
            "entries": list(self.entries),
            "metadata": {**self.metadata, **dict(metadata or {})},
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, *, expected_key_dim: int | None = None, expected_value_dim: int | None = None) -> "TaskContextBank":
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("format") != cls.FORMAT:
            raise ValueError(f"Unsupported task-context bank format: {path}")
        config = dict(payload.get("config", {}))
        key_dim = int(config.get("key_dim", 256))
        value_dim = int(config.get("value_dim", 256))
        if expected_key_dim is not None and key_dim != int(expected_key_dim):
            raise ValueError(f"task-context bank key_dim={key_dim} does not match expected {expected_key_dim}")
        if expected_value_dim is not None and value_dim != int(expected_value_dim):
            raise ValueError(f"task-context bank value_dim={value_dim} does not match expected {expected_value_dim}")
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise ValueError("task-context bank entries must be a list")
        return cls(entries, key_dim=key_dim, value_dim=value_dim, temperature=float(config.get("temperature", 0.07)), metadata=payload.get("metadata", {}))


def retrieve_task_context(
    context: Tensor,
    context_mask: Tensor | None,
    bank: TaskContextBank,
    *,
    output_dim: int,
    top_k: int = 1,
) -> tuple[Tensor, TaskContextRetrievalResult]:
    """Return [B,V] prototypes, including B=1 for an unbatched text context."""
    if bank.value_dim != int(output_dim):
        raise ValueError(f"bank behavior_value dim {bank.value_dim} does not match output_dim {output_dim}")
    if context.ndim == 2:
        context = context.unsqueeze(0)
    if context.ndim != 3:
        raise ValueError("context must be [B,L,D] or [L,D]")
    if context_mask is not None and context_mask.ndim == 1:
        context_mask = context_mask.unsqueeze(0)
    query = task_tokens_from_context(context, context_mask, bank.key_dim)
    result = bank.retrieve(query, top_k=top_k)
    return result.values, result
