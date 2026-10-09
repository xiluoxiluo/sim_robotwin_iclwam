"""Validate and materialize the 32-action -> 8-transition RoboTwin view."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from fastwam.datasets.zeva_robotwin_dataset import ZevaRobotWinDataset
from fastwam.zeva.schemas import sha256_file


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    stats_path = str(cfg.data.train.get("pretrained_norm_stats", ""))
    if stats_path in {"", "None", "null"} or not Path(stats_path).is_file():
        raise FileNotFoundError(
            "Zeva transition materialization requires an existing data.train.pretrained_norm_stats file"
        )
    base = instantiate(cfg.data.train)
    dataset = ZevaRobotWinDataset(base)
    if int(cfg.data.train.get("global_sample_stride", 1)) != 1:
        raise ValueError(
            "RoboTwin Zeva V1 requires data.train.global_sample_stride=1 for exact "
            f"frame/action alignment; got {cfg.data.train.global_sample_stride}"
        )
    if int(cfg.data.train.action_video_freq_ratio) != 4:
        raise ValueError(
            "RoboTwin Zeva V1 requires action_video_freq_ratio=4; "
            f"got {cfg.data.train.action_video_freq_ratio}"
        )
    output = Path(str(cfg.model.get("zeva", {}).get("transition_path", "./data/robotwin2.0/zeva_transitions")))
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    seen_indices = set()
    for index in range(len(dataset)):
        sample = dataset[index]
        episode = sample["episode"]
        source_index = int(sample.get("dataset_index", index))
        if source_index in seen_indices:
            raise ValueError(f"dataset returned duplicate source index {source_index}; refusing to overwrite transition file")
        seen_indices.add(source_index)
        torch.save({key: sample[key] for key in ("transition_actions", "before_frames", "after_frames", "transition_valid", "frame_valid", "action_valid")}, output / f"{source_index:08d}.pt")
        rows.append({"index": source_index, "episode_id": episode.episode_id, "task_id": episode.task_id, "transition_count": 8})
    stats_hash = sha256_file(stats_path) if stats_path and stats_path not in {"None", "null"} else ""
    manifest = {
        "schema_version": "zeva_fastwam_robotwin_transitions_v1",
        "dataset_path": str(cfg.data.train.dataset_dirs[0]),
        "dataset_stats_sha256": stats_hash,
        "camera_keys": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "action_dim": 14,
        "action_horizon": 32,
        "action_video_freq_ratio": int(cfg.data.train.action_video_freq_ratio),
        "action_normalization": "fastwam_processor_output",
        "transition_count": 8,
        "rows": rows,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(rows)} episode-aware transition windows to {output}")


if __name__ == "__main__":
    main()
