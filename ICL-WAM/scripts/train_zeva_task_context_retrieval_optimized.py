"""Train and validate the Zeva static task-context retrieval head.

This keeps Zeva's symmetric supervised contrastive objective unchanged while
adding strict artifact/semantic checks and a final held-out retrieval report.
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.nn import functional as F

from fastwam.zeva import TaskContextBank
from fastwam.zeva.checkpoint import checkpoint_sha256
from fastwam.zeva.static_task_context import (
    HEAD_FORMAT,
    READOUT_KIND,
    StaticTaskContextRetrievalConfig,
    StaticTaskContextRetrievalHead,
    load_readout_cache,
    validate_behavior_bank,
)
from fastwam.zeva.task_context_training import train_retrieval_head
from fastwam.zeva.schemas import sha256_file


def _full_validation(head, readouts, keys, labels, train_indices, val_indices, device):
    head.eval()
    readouts = readouts.float().to(device)
    keys = F.normalize(keys.float().to(device), dim=-1)
    labels = labels.to(device)
    train_idx = torch.tensor(train_indices, device=device, dtype=torch.long)
    val_idx = torch.tensor(val_indices, device=device, dtype=torch.long)
    with torch.no_grad():
        q = head(readouts[val_idx])
        scores = q @ keys[train_idx].T
        nearest = scores.argmax(dim=-1)
        predicted = labels[train_idx][nearest]
        top1 = (predicted == labels[val_idx]).float().mean().item()
        top5_idx = scores.topk(min(5, scores.shape[1]), dim=-1).indices
        top5_labels = labels[train_idx][top5_idx]
        top5 = top5_labels.eq(labels[val_idx, None]).any(dim=-1).float().mean().item()
        same = labels[val_idx, None].eq(labels[train_idx][None, :])
        same_scores = scores.masked_fill(~same, torch.nan)
        cross_scores = scores.masked_fill(same, torch.nan)
        same_mean = torch.nanmean(same_scores).item()
        cross_mean = torch.nanmean(cross_scores).item()
    return {
        "heldout_task_top1": float(top1),
        "heldout_task_top5": float(top5),
        "heldout_same_task_cosine_mean": float(same_mean),
        "heldout_cross_task_cosine_mean": float(cross_mean),
    }


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    tc = cfg.model.zeva.task_context
    for name in ("bank_path", "readout_cache_path", "retrieval_checkpoint"):
        value = tc.get(name)
        if value in (None, "", "None", "null"):
            raise ValueError(f"Set model.zeva.task_context.{name}")
    bank_path = Path(str(tc.bank_path)).expanduser().resolve()
    readout_path = Path(str(tc.readout_cache_path)).expanduser().resolve()
    head_path = Path(str(tc.retrieval_checkpoint)).expanduser().resolve()
    if head_path in {bank_path, readout_path}:
        raise ValueError("retrieval checkpoint must not overwrite the bank/readout cache")
    if not bank_path.is_file() or not readout_path.is_file():
        raise FileNotFoundError("behavior bank/readout cache must be built first")

    bank = TaskContextBank.load(bank_path)
    validate_behavior_bank(bank)
    semantic_path = Path(str(cfg.data.train.semantic_task_map_path)).expanduser().resolve()
    stats_path = Path(str(cfg.data.train.pretrained_norm_stats)).expanduser().resolve()
    if not semantic_path.is_file() or not stats_path.is_file():
        raise FileNotFoundError("semantic map and dataset stats must exist for retrieval training")
    expected_bank_identity = {
        "semantic_task_map_sha256": sha256_file(semantic_path),
        "dataset_stats_sha256": sha256_file(stats_path),
        "video_size": [int(v) for v in cfg.data.train.video_size],
        "context_len": int(cfg.model.tokenizer_max_len),
        "history_semantics": "full_episode_prefix",
        "semantic_task_count": 50,
    }
    for key, expected in expected_bank_identity.items():
        if bank.metadata.get(key) != expected:
            raise ValueError(
                f"behavior-bank source identity mismatch for {key}: "
                f"expected={expected!r}, actual={bank.metadata.get(key)!r}"
            )
    if len(bank) < 2:
        raise ValueError("static retrieval training requires at least two demonstrations")
    task_names = sorted({str(entry["task_id"]) for entry in bank.entries})
    if len(task_names) != 50:
        raise ValueError(f"formal RoboTwin bank must contain 50 semantic tasks, got {len(task_names)}")
    counts = {task: 0 for task in task_names}
    for entry in bank.entries:
        counts[str(entry["task_id"])] += 1
    if min(counts.values()) < 2:
        raise ValueError("every semantic task requires >=2 demonstrations for disjoint validation")

    cache = load_readout_cache(readout_path, bank_path, bank)
    readouts = cache["readouts"]
    keys = torch.stack([entry["retrieval_key"] for entry in bank.entries]).float()
    if keys.shape != (len(bank), bank.key_dim):
        raise ValueError("behavior-bank retrieval-key matrix shape mismatch")
    if readouts.dtype != torch.float32:
        readouts = readouts.float()
    if not bool(torch.isfinite(readouts).all() and torch.isfinite(keys).all()):
        raise FloatingPointError("retrieval training inputs contain NaN/Inf")

    head_cfg = StaticTaskContextRetrievalConfig(**dict(tc.retrieval))
    if head_cfg.input_dim != readouts.shape[1]:
        raise ValueError(
            f"retrieval head input_dim={head_cfg.input_dim} != readout_dim={readouts.shape[1]}"
        )
    if head_cfg.output_dim != bank.key_dim:
        raise ValueError(
            f"retrieval head output_dim={head_cfg.output_dim} != bank.key_dim={bank.key_dim}"
        )

    task_to_index = {task: index for index, task in enumerate(task_names)}
    labels = torch.tensor([task_to_index[str(entry["task_id"])] for entry in bank.entries], dtype=torch.long)
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    configured_device = cfg.get("device")
    device = torch.device(str(configured_device) if configured_device else ("cuda" if torch.cuda.is_available() else "cpu"))
    head = StaticTaskContextRetrievalHead(head_cfg).to(device)

    metrics = train_retrieval_head(
        head,
        readouts,
        keys,
        labels,
        seed=seed,
        **dict(tc.training),
    )
    final_report = _full_validation(
        head,
        readouts,
        keys,
        labels,
        metrics["train_indices"],
        metrics["validation_indices"],
        device,
    )
    if not all(torch.isfinite(torch.tensor(value)) for value in final_report.values()):
        raise FloatingPointError("static retrieval validation produced non-finite metrics")

    head_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": HEAD_FORMAT,
        "readout_kind": READOUT_KIND,
        "bank_sha256": checkpoint_sha256(bank_path),
        "readout_cache_sha256": checkpoint_sha256(readout_path),
        "config": head_cfg.as_dict(),
        "model": {k: v.cpu() for k, v in head.state_dict().items()},
        "step": metrics["step"],
        "training": dict(tc.training),
        "seed": seed,
        "metrics": metrics["history"],
        "validation": final_report,
        "semantic_tasks": task_names,
        "train_episode_ids": [cache["episode_ids"][i] for i in metrics["train_indices"]],
        "validation_episode_ids": [cache["episode_ids"][i] for i in metrics["validation_indices"]],
    }
    tmp_head = head_path.with_name(f".{head_path.name}.tmp")
    torch.save(payload, tmp_head)
    # Verify the serialized checkpoint before publishing it.
    reloaded = torch.load(tmp_head, map_location="cpu", weights_only=False)
    if reloaded.get("format") != HEAD_FORMAT or reloaded.get("bank_sha256") != payload["bank_sha256"]:
        raise RuntimeError("serialized static retrieval checkpoint failed identity validation")
    tmp_head.replace(head_path)
    report_path = head_path.with_suffix(head_path.suffix + ".validation.json")
    report_path.write_text(json.dumps(final_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("========== static retrieval head ==========")
    print(f"best step: {metrics['step']}")
    for key, value in final_report.items():
        print(f"{key}: {value:.6f}")
    print(f"checkpoint: {head_path}")
    print("STATIC RETRIEVAL HEAD: PASSED")


if __name__ == "__main__":
    main()
