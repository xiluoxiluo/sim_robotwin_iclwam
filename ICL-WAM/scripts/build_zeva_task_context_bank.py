"""Build a FastWAM-specific static task-context prototype bank.

This first bank builder uses the same precomputed FastWAM text context as
Stage 2.  It creates one prototype per ``(task_id, instruction)`` and keeps
the artifact format independent from the online BIT/PIM memory.  The resulting
artifact is a deterministic bootstrap bank; a learned FastWAM retrieval head
can later replace the context pooling used to produce its keys and values.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from fastwam.datasets.zeva_robotwin_dataset import ZevaRobotWinDataset
from fastwam.zeva import TaskContextBank, task_tokens_from_context


@torch.no_grad()
def build_bank(base_dataset, *, key_dim: int, value_dim: int, temperature: float = 0.07) -> TaskContextBank:
    """Average sample contexts per task/instruction with bounded accumulator memory."""
    dataset = ZevaRobotWinDataset(base_dataset)
    grouped: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor, int]] = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        context = sample["context"].unsqueeze(0)
        context_mask = sample.get("context_mask")
        if context_mask is not None:
            context_mask = context_mask.unsqueeze(0)
        key = task_tokens_from_context(context, context_mask, key_dim)[0].float().cpu()
        value = task_tokens_from_context(context, context_mask, value_dim)[0].float().cpu()
        group = (str(sample["episode"].task_id), sample["episode"].instruction)
        if group in grouped:
            key_sum, value_sum, count = grouped[group]
            grouped[group] = (key_sum + key, value_sum + value, count + 1)
        else:
            grouped[group] = (key, value, 1)
    entries = []
    for (task_id, instruction), (key_sum, value_sum, count) in sorted(grouped.items()):
        entries.append({
            "retrieval_key": key_sum / count,
            "behavior_value": value_sum / count,
            "task_id": task_id,
            "instruction": instruction,
            "num_samples": count,
        })
    return TaskContextBank(entries, key_dim=key_dim, value_dim=value_dim, temperature=temperature)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if cfg.model is None or cfg.data is None:
        raise ValueError("Select task=robotwin_zeva_fastwam_3cam_384 for bank construction")
    task_context_cfg = cfg.model.get("zeva", {}).get("task_context", {})
    output = task_context_cfg.get("bank_path")
    if output in (None, "", "None", "null"):
        raise ValueError("Set model.zeva.task_context.bank_path for bank construction")
    key_dim = int(task_context_cfg.get("key_dim", 256))
    value_dim = int(task_context_cfg.get("value_dim", cfg.model.get("zeva", {}).get("prompt", {}).get("global_dim", 256)))
    prompt_dim = int(cfg.model.zeva.prompt.global_dim)
    if value_dim != prompt_dim:
        raise ValueError("zeva.task_context.value_dim must match zeva.prompt.global_dim")
    base = instantiate(cfg.data.train)
    bank = build_bank(
        base, key_dim=key_dim, value_dim=value_dim,
        temperature=float(task_context_cfg.get("temperature", 0.07)),
    )
    output_path = Path(str(output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bank.save(output_path, metadata={
        "builder": "build_zeva_task_context_bank.py",
        "source": "FastWAM precomputed text context",
        "prototype_group": "task_id,instruction",
        "num_source_samples": len(base),
    })
    print(f"Wrote {len(bank)} task-context prototypes to {output_path}")


if __name__ == "__main__":
    main()
