# SPDX-FileCopyrightText: Copyright (c) 2026 Z-Trans CORPORATION. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Zeva static retrieval architecture/loss with FastWAM artifact contracts.

The head and symmetric multi-positive loss follow Zeva's
cosmos_framework/model/zeva/static_task_context_retrieval.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .checkpoint import checkpoint_sha256
from .task_context import TaskContextBank, TaskContextRetrievalResult


READOUT_KIND = "fastwam_clean_initial_video_hidden_mean_v1"
BEHAVIOR_KEY_SPACE = "cte_trajectory_retrieval_mean_v1"
BEHAVIOR_VALUE_SPACE = "cte_trajectory_state_mean_v1"
HEAD_FORMAT = "fastwam_static_task_context_head_v1"
READOUT_FORMAT = "fastwam_static_task_context_readouts_v1"


@dataclass(frozen=True)
class StaticTaskContextRetrievalConfig:
    input_dim: int = 3072
    hidden_dim: int = 1024
    output_dim: int = 128
    dropout: float = 0.1

    def __post_init__(self):
        if min(self.input_dim, self.hidden_dim, self.output_dim) < 1:
            raise ValueError("retrieval dimensions must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("retrieval dropout must be in [0,1)")

    def as_dict(self) -> dict:
        return asdict(self)


class StaticTaskContextRetrievalHead(nn.Module):
    def __init__(self, config: StaticTaskContextRetrievalConfig):
        super().__init__()
        self.config = config
        self.projection = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(self, readout: Tensor) -> Tensor:
        if readout.ndim != 2 or readout.shape[-1] != self.config.input_dim:
            raise ValueError(f"Expected readout [B,{self.config.input_dim}]")
        return F.normalize(self.projection(readout), dim=-1)


def bidirectional_supervised_contrastive_loss(
    queries: Tensor, keys: Tensor, semantic_ids: Tensor, *, temperature: float = 0.07,
) -> Tensor:
    """Zeva's symmetric multi-positive InfoNCE, with fixed CTE target keys."""
    if queries.ndim != 2 or keys.shape != queries.shape:
        raise ValueError("Expected aligned query/key [B,D] tensors")
    if semantic_ids.shape != (queries.shape[0],):
        raise ValueError("semantic_ids must contain one task ID per pair")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    queries, keys = F.normalize(queries.float(), dim=-1), F.normalize(keys.float(), dim=-1)
    logits = queries @ keys.T / temperature
    positive = semantic_ids[:, None].eq(semantic_ids[None, :])

    def direction(scores, mask):
        return (
            torch.logsumexp(scores, dim=1)
            - torch.logsumexp(scores.masked_fill(~mask, -torch.inf), dim=1)
        ).mean()

    return 0.5 * (direction(logits, positive) + direction(logits.T, positive.T))


def trajectory_prototype(outputs: dict[str, Tensor], valid: Tensor) -> tuple[Tensor, Tensor]:
    """CTE trajectory key and behavior state, each pooled over valid states."""
    retrieval, states = outputs["retrieval"], outputs["causal_interaction_state"]
    if retrieval.ndim != 3 or states.ndim != 3 or retrieval.shape[:2] != states.shape[:2]:
        raise ValueError("CTE retrieval/state tensors must share [B,T]")
    if valid.shape != retrieval.shape[:2] or not bool(valid.bool().any(dim=1).all()):
        raise ValueError("Every trajectory needs at least one valid CTE state")
    mask = valid.bool().unsqueeze(-1)
    count = mask.sum(dim=1).float()
    key = retrieval.float().masked_fill(~mask, 0).sum(dim=1) / count
    value = states.float().masked_fill(~mask, 0).sum(dim=1) / count
    return F.normalize(key, dim=-1), value


def validate_behavior_bank(bank: TaskContextBank, expected: dict | None = None) -> None:
    for name, value in {
        "key_space": BEHAVIOR_KEY_SPACE,
        "value_space": BEHAVIOR_VALUE_SPACE,
        "readout_kind": READOUT_KIND,
        "split": "train",
        **dict(expected or {}),
    }.items():
        if bank.metadata.get(name) != value:
            raise ValueError(f"task-context bank {name} mismatch: expected {value!r}, got {bank.metadata.get(name)!r}")
    ids = [entry.get("episode_id") for entry in bank.entries]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("behavior bank requires unique source episode IDs")


class StaticTaskContextRetriever:
    """Frozen task prior, independent of the online phase/effect memory."""

    def __init__(self, bank: TaskContextBank, head: StaticTaskContextRetrievalHead, *, top_k: int = 5):
        validate_behavior_bank(bank)
        if head.config.output_dim != bank.key_dim:
            raise ValueError("static retrieval head output_dim does not match bank key_dim")
        if head.config.input_dim != bank.metadata.get("readout_dim"):
            raise ValueError("static retrieval head input_dim does not match the bank readout_dim")
        if top_k < 1:
            raise ValueError("static retrieval top_k must be positive")
        self.bank, self.head = bank, head.eval().requires_grad_(False)
        # Zeva's serving path clips K to the number of available examples.
        self.top_k = min(int(top_k), len(bank))

    @torch.no_grad()
    def retrieve(self, readout: Tensor, *, exclude_episode_ids=None) -> TaskContextRetrievalResult:
        parameter = next(self.head.parameters())
        query = self.head(readout.to(device=parameter.device, dtype=parameter.dtype))
        top_k = self.top_k
        if exclude_episode_ids is not None:
            top_k = min(top_k, len(self.bank) - 1)
            if top_k < 1:
                raise ValueError("leave-one-episode-out retrieval needs at least two demonstrations")
        return self.bank.retrieve(query, top_k=top_k, exclude_episode_ids=exclude_episode_ids)

    @classmethod
    def load(cls, bank_path, head_path, *, top_k=5, device="cpu", expected=None):
        bank = TaskContextBank.load(bank_path)
        validate_behavior_bank(bank, expected)
        payload = torch.load(head_path, map_location="cpu", weights_only=False)
        if payload.get("format") != HEAD_FORMAT or int(payload.get("step", 0)) < 1:
            raise ValueError("static retrieval requires a trained FastWAM head checkpoint")
        if payload.get("bank_sha256") != checkpoint_sha256(bank_path):
            raise ValueError("static retrieval head/bank identity mismatch")
        if payload.get("readout_kind") != READOUT_KIND:
            raise ValueError("static retrieval readout kind mismatch")
        head = StaticTaskContextRetrievalHead(StaticTaskContextRetrievalConfig(**payload["config"]))
        head.load_state_dict(payload["model"], strict=True)
        head.to(device)
        return cls(bank, head, top_k=top_k)


def load_readout_cache(path, bank_path, bank: TaskContextBank) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != READOUT_FORMAT or payload.get("readout_kind") != READOUT_KIND:
        raise ValueError("unsupported FastWAM initial-readout cache")
    if payload.get("bank_sha256") != checkpoint_sha256(bank_path):
        raise ValueError("readout cache/bank identity mismatch")
    expected_ids = [entry["episode_id"] for entry in bank.entries]
    if payload.get("episode_ids") != expected_ids:
        raise ValueError("readout cache episode order does not match the behavior bank")
    readouts = payload.get("readouts")
    if not isinstance(readouts, Tensor) or readouts.ndim != 2 or readouts.shape[0] != len(bank):
        raise ValueError("readout cache must contain [N,input_dim]")
    if not bool(torch.isfinite(readouts).all()):
        raise ValueError("readout cache contains non-finite values")
    return payload


@torch.no_grad()
def stage2_task_contexts(retriever: StaticTaskContextRetriever, cache: dict) -> dict[str, Tensor]:
    result = {}
    for episode_id, readout in zip(cache["episode_ids"], cache["readouts"], strict=True):
        retrieval = retriever.retrieve(readout.unsqueeze(0), exclude_episode_ids=[episode_id])
        result[episode_id] = retrieval.values[0].cpu()
    return result


def task_context_identity(mode: str, bank_path, head_path=None, top_k=5) -> dict:
    return {
        "mode": mode, "bank_sha256": checkpoint_sha256(bank_path),
        "head_sha256": None if head_path is None else checkpoint_sha256(head_path),
        "top_k": int(top_k), "readout_kind": READOUT_KIND,
    }


class StaticTaskContextSession:
    """One fixed prior per attempt, computed only from its initial observation."""

    def __init__(self, retriever: StaticTaskContextRetriever):
        self.retriever = retriever
        self.reset()

    def reset(self):
        self.result = None
        self.instruction = None

    @torch.no_grad()
    def resolve(self, policy, image, context, context_mask, instruction: str):
        if self.result is None:
            readout = policy.extract_task_context_readout(image, context, context_mask)
            self.result = self.retriever.retrieve(readout)
            self.instruction = instruction
        elif instruction != self.instruction:
            raise ValueError("task instruction changed within an attempt; reset task-context session first")
        return self.result
