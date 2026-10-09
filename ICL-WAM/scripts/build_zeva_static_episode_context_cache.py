"""Precompute leave-one-episode-out static task context for Stage-2 training.

Online evaluation still computes the FastWAM initial readout and retrieves from
bank/head on the first observation.  Training already owns one frozen readout
per demonstration, so recomputing N separate full-bank searches every process
startup is wasteful.  This script vectorizes exactly the same top-K cosine +
softmax rule and stores only the resulting episode-level [256] values.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.nn import functional as F
from tqdm import tqdm

from fastwam.zeva import TaskContextBank
from fastwam.zeva.checkpoint import checkpoint_sha256
from fastwam.zeva.prepared_stage2 import STATIC_CONTEXT_FORMAT
from fastwam.zeva.schemas import sha256_file
from fastwam.zeva.static_task_context import (
    StaticTaskContextRetriever,
    load_readout_cache,
    validate_behavior_bank,
)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(default if value in (None, "") else value)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    tc = cfg.model.zeva.task_context
    for name in ("bank_path", "readout_cache_path", "retrieval_checkpoint", "static_episode_cache_path"):
        if tc.get(name) in (None, "", "None", "null"):
            raise ValueError(f"Set model.zeva.task_context.{name}")
    bank_path = Path(str(tc.bank_path)).expanduser().resolve()
    readout_path = Path(str(tc.readout_cache_path)).expanduser().resolve()
    head_path = Path(str(tc.retrieval_checkpoint)).expanduser().resolve()
    output_path = Path(str(tc.static_episode_cache_path)).expanduser().resolve()
    for path in (bank_path, readout_path, head_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists() and os.environ.get("ZEVA_STATIC_CONTEXT_OVERWRITE", "0") not in {"1", "true", "TRUE"}:
        raise FileExistsError(f"static episode-context cache exists: {output_path}")

    bank = TaskContextBank.load(bank_path)
    semantic_path = Path(str(cfg.data.train.semantic_task_map_path)).expanduser().resolve()
    stats_path = Path(str(cfg.data.train.pretrained_norm_stats)).expanduser().resolve()
    if not semantic_path.is_file() or not stats_path.is_file():
        raise FileNotFoundError("semantic map and dataset stats must exist for static-context build")
    expected_bank_identity = {
        "semantic_task_map_sha256": sha256_file(semantic_path),
        "dataset_stats_sha256": sha256_file(stats_path),
        "video_size": [int(v) for v in cfg.data.train.video_size],
        "context_len": int(cfg.model.tokenizer_max_len),
        "history_semantics": "full_episode_prefix",
        "semantic_task_count": 50,
    }
    validate_behavior_bank(bank, expected_bank_identity)
    readout_payload = load_readout_cache(readout_path, bank_path, bank)
    retriever = StaticTaskContextRetriever.load(
        bank_path,
        head_path,
        top_k=int(tc.top_k),
        device="cuda" if torch.cuda.is_available() else "cpu",
        expected=expected_bank_identity,
    )
    device = next(retriever.head.parameters()).device
    readouts = readout_payload["readouts"].float()
    episode_ids = list(readout_payload["episode_ids"])
    if len(episode_ids) != len(bank):
        raise ValueError("readout/bank episode count mismatch")

    keys = torch.stack([entry["retrieval_key"] for entry in bank.entries]).float().to(device)
    values = torch.stack([entry["behavior_value"] for entry in bank.entries]).float().to(device)
    keys = F.normalize(keys, dim=-1)
    top_k = int(tc.top_k)
    if not 1 <= top_k < len(bank):
        raise ValueError("leave-one-out static retrieval requires 1 <= top_k < bank size")
    batch_size = _env_int("ZEVA_STATIC_CONTEXT_BATCH_SIZE", 512)

    contexts = torch.empty((len(bank), bank.value_dim), dtype=torch.float32)
    indices_all = torch.empty((len(bank), top_k), dtype=torch.int64)
    scores_all = torch.empty((len(bank), top_k), dtype=torch.float32)
    iterator = range(0, len(bank), batch_size)
    progress = tqdm(iterator, total=(len(bank) + batch_size - 1) // batch_size, desc="static context", unit="batch")
    with torch.inference_mode():
        for start in progress:
            end = min(len(bank), start + batch_size)
            q = retriever.head(readouts[start:end].to(device))
            scores = q @ keys.T
            local_rows = torch.arange(end - start, device=device)
            global_rows = torch.arange(start, end, device=device)
            scores[local_rows, global_rows] = -torch.inf
            top_scores, top_indices = torch.topk(scores, top_k, dim=-1, largest=True, sorted=True)
            weights = F.softmax(top_scores / float(bank.temperature), dim=-1)
            selected = values[top_indices]
            context = torch.einsum("bk,bkd->bd", weights, selected)
            contexts[start:end] = context.float().cpu()
            indices_all[start:end] = top_indices.cpu()
            scores_all[start:end] = top_scores.float().cpu()

    if not bool(torch.isfinite(contexts).all() and torch.isfinite(scores_all).all()):
        raise FloatingPointError("static episode-context cache contains NaN/Inf")

    # Random deterministic equivalence test against the public online retriever.
    validation_samples = min(_env_int("ZEVA_STATIC_CONTEXT_VALIDATE", 64), len(bank))
    generator = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
    sample_indices = torch.randperm(len(bank), generator=generator)[:validation_samples].tolist()
    worst_value = 0.0
    worst_score = 0.0
    for index in sample_indices:
        result = retriever.retrieve(
            readouts[index].unsqueeze(0),
            exclude_episode_ids=[episode_ids[index]],
        )
        value_diff = (result.values[0].cpu().float() - contexts[index]).abs().max().item()
        score_diff = (result.scores[0].cpu().float() - scores_all[index]).abs().max().item()
        if not torch.equal(result.indices[0].cpu(), indices_all[index]):
            raise RuntimeError(f"static retrieval indices mismatch at episode {episode_ids[index]}")
        worst_value = max(worst_value, value_diff)
        worst_score = max(worst_score, score_diff)
    tolerance = 1.0e-5
    if worst_value > tolerance or worst_score > tolerance:
        raise RuntimeError(
            f"vectorized static retrieval exceeds tolerance={tolerance}: "
            f"value={worst_value}, score={worst_score}"
        )

    metadata = {
        **dict(bank.metadata),
        "bank_sha256": checkpoint_sha256(bank_path),
        "retrieval_checkpoint_sha256": checkpoint_sha256(head_path),
        "readout_cache_sha256": checkpoint_sha256(readout_path),
        "top_k": top_k,
        "temperature": float(bank.temperature),
        "validation_samples": validation_samples,
        "validation_worst_value_abs": worst_value,
        "validation_worst_score_abs": worst_score,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": STATIC_CONTEXT_FORMAT,
        "episode_ids": episode_ids,
        "task_contexts": contexts,
        "retrieval_indices": indices_all,
        "retrieval_scores": scores_all,
        "metadata": metadata,
    }
    tmp_output = output_path.with_name(f".{output_path.name}.tmp")
    torch.save(payload, tmp_output)
    verify = torch.load(tmp_output, map_location="cpu", weights_only=False)
    if verify.get("format") != STATIC_CONTEXT_FORMAT or verify.get("episode_ids") != episode_ids:
        raise RuntimeError("serialized static episode-context cache failed validation")
    tmp_output.replace(output_path)
    print("========== static episode context ==========")
    print(f"episodes: {len(episode_ids):,}")
    print(f"context shape: {tuple(contexts.shape)}")
    print(f"top_k: {top_k}")
    print(f"equivalence worst abs: value={worst_value}, score={worst_score}")
    print(f"output: {output_path}")
    print("STATIC EPISODE CONTEXT CACHE: PASSED")


if __name__ == "__main__":
    main()
