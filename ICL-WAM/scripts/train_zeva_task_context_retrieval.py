"""Train Zeva's static retrieval head on cached FastWAM initial readouts."""

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from fastwam.zeva import TaskContextBank
from fastwam.zeva.checkpoint import checkpoint_sha256
from fastwam.zeva.static_task_context import (
    HEAD_FORMAT, READOUT_KIND, StaticTaskContextRetrievalConfig,
    StaticTaskContextRetrievalHead, load_readout_cache, validate_behavior_bank,
)
from fastwam.zeva.task_context_training import train_retrieval_head


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    tc = cfg.model.zeva.task_context
    for name in ("bank_path", "readout_cache_path", "retrieval_checkpoint"):
        if tc.get(name) in (None, "", "None", "null"):
            raise ValueError(f"Set model.zeva.task_context.{name}")
    bank = TaskContextBank.load(str(tc.bank_path))
    if Path(str(tc.retrieval_checkpoint)).resolve() in {
        Path(str(tc.bank_path)).resolve(), Path(str(tc.readout_cache_path)).resolve(),
    }:
        raise ValueError("retrieval checkpoint must not overwrite the bank or readout cache")
    validate_behavior_bank(bank)
    cache = load_readout_cache(str(tc.readout_cache_path), str(tc.bank_path), bank)
    readouts = cache["readouts"]
    head_config = StaticTaskContextRetrievalConfig(**dict(tc.retrieval))
    if head_config.input_dim != readouts.shape[1] or head_config.output_dim != bank.key_dim:
        raise ValueError("retrieval head dimensions do not match readouts and CTE keys")
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    configured_device = cfg.get("device")
    device = str(configured_device) if configured_device else ("cuda" if torch.cuda.is_available() else "cpu")
    head = StaticTaskContextRetrievalHead(head_config).to(device)
    entries = bank.entries
    tasks = {task: index for index, task in enumerate(sorted({str(entry["task_id"]) for entry in entries}))}
    labels = torch.tensor([tasks[str(entry["task_id"])] for entry in entries])
    keys = torch.stack([entry["retrieval_key"] for entry in entries])
    metrics = train_retrieval_head(head, readouts, keys, labels, seed=seed, **dict(tc.training))
    path = Path(str(tc.retrieval_checkpoint))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": HEAD_FORMAT, "readout_kind": READOUT_KIND,
        "bank_sha256": checkpoint_sha256(str(tc.bank_path)),
        "readout_cache_sha256": checkpoint_sha256(str(tc.readout_cache_path)),
        "config": head_config.as_dict(), "model": {k: v.cpu() for k, v in head.state_dict().items()},
        "step": metrics["step"], "training": dict(tc.training), "seed": seed,
        "metrics": metrics["history"],
        "train_episode_ids": [cache["episode_ids"][i] for i in metrics["train_indices"]],
        "validation_episode_ids": [cache["episode_ids"][i] for i in metrics["validation_indices"]],
    }, path)
    print(f"Saved trained static task-context retrieval head to {path}")


if __name__ == "__main__":
    main()
