"""Episode-ordered phase/effect cache and deterministic shard IO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import load_file, save_file

from .schemas import CacheManifest


def _validate_v4_temporal_row(row: dict, index: int) -> None:
    record_type = row.get("record_type")
    if record_type not in {"phase_query", "effect"}:
        raise ValueError(f"cache row {index} has unsupported v4 record_type={record_type!r}")
    required = ("episode_step", "raw_step", "start_raw_step", "end_raw_step")
    if any(row.get(key) is None for key in required):
        raise ValueError(f"cache row {index} has incomplete v4 temporal metadata")
    try:
        episode_step = int(row["episode_step"])
        raw_step = int(row["raw_step"])
        start_raw_step = int(row["start_raw_step"])
        end_raw_step = int(row["end_raw_step"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cache row {index} has non-integer v4 temporal metadata") from exc
    if min(episode_step, raw_step, start_raw_step, end_raw_step) < 0:
        raise ValueError(f"cache row {index} has negative v4 temporal metadata")
    if start_raw_step > end_raw_step:
        raise ValueError(f"cache row {index} has start_raw_step after end_raw_step")
    if record_type == "phase_query":
        if row.get("window_index") is None:
            raise ValueError(f"cache row {index} phase_query must have window_index")
        if not (episode_step == raw_step == start_raw_step == end_raw_step):
            raise ValueError(
                f"cache row {index} phase_query must identify one raw boundary; "
                f"got episode_step={episode_step}, raw_step={raw_step}, "
                f"start={start_raw_step}, end={end_raw_step}"
            )
    elif row.get("window_index") is not None:
        raise ValueError(f"cache row {index} effect must not have window_index")
    if record_type == "effect" and start_raw_step == end_raw_step:
        raise ValueError(f"cache row {index} effect must cover a positive raw-step interval")
    if record_type == "effect" and (episode_step != start_raw_step or raw_step != end_raw_step):
        raise ValueError(
            f"cache row {index} effect must use episode_step/start and raw_step/end; "
            f"got episode_step={episode_step}, raw_step={raw_step}, "
            f"start={start_raw_step}, end={end_raw_step}"
        )


class PhaseEffectCache:
    def __init__(self, root: str | Path, manifest: CacheManifest, rows: list[dict]):
        self.root, self.manifest, self.rows = Path(root), manifest, rows
        self._shards: dict[str, dict[str, torch.Tensor]] = {}
        required = {"episode_id", "task_id", "attempt_id", "transition_index", "window_index", "episode_step", "shard", "offset"}
        if manifest.schema_version in {"zeva_fastwam_robotwin_cache_v3", "zeva_fastwam_robotwin_cache_v4"}:
            required.add("effect_index")
        if manifest.schema_version == "zeva_fastwam_robotwin_cache_v4":
            required.update({"record_type", "raw_step", "start_raw_step", "end_raw_step"})
        for index, row in enumerate(rows):
            missing = required - set(row)
            if missing:
                raise ValueError(f"cache row {index} missing fields: {sorted(missing)}")
            if manifest.schema_version == "zeva_fastwam_robotwin_cache_v4":
                _validate_v4_temporal_row(row, index)
            shard = self.root / str(row["shard"])
            if not shard.is_file():
                raise FileNotFoundError(f"cache row {index} references missing shard: {shard}")

    def __len__(self) -> int:
        return len(self.rows)

    def _load_shard(self, name: str) -> dict[str, torch.Tensor]:
        # Stage 2 repeatedly reads the same ordered shards while constructing
        # BIT and the offline memory bank. Cache the opened tensor map per
        # process instead of reopening a safetensors file for every row.
        if name not in self._shards:
            self._shards[name] = load_file(str(self.root / name), device="cpu")
        return self._shards[name]

    def get(self, index: int) -> dict:
        row = self.rows[int(index)]
        shard = self._load_shard(str(row["shard"]))
        offset = int(row["offset"])
        return {
            "episode_id": row["episode_id"], "task_id": row["task_id"],
            "attempt_id": row["attempt_id"], "transition_index": row["transition_index"],
            # Legacy v2 rows used one record per transition and have no
            # effect_index. Preserve that index for backwards-compatible
            # loading; v3/v4 rows write effect_index explicitly.
            "effect_index": int(row.get("effect_index", row["transition_index"])),
            "record_type": row.get("record_type", "effect"),
            "episode_step": row.get("episode_step"),
            "raw_step": row.get("raw_step"),
            "start_raw_step": row.get("start_raw_step"),
            "end_raw_step": row.get("end_raw_step"),
            "window_index": row.get("window_index"),
            "phase_pre": shard["phase_pre"][offset], "phase_post": shard["phase_post"][offset],
            "effect": shard["effect"][offset], "valid": bool(shard["valid"][offset].item()),
        }

    @classmethod
    def load(cls, root: str | Path, expected: dict[str, object] | None = None) -> "PhaseEffectCache":
        root = Path(root)
        manifest = CacheManifest(**json.loads((root / "manifest.json").read_text()))
        manifest.validate(expected)
        rows = json.loads((root / "episode_index.json").read_text())
        return cls(root, manifest, rows)


def save_phase_effect_cache(root: str | Path, records: Iterable[dict], manifest: CacheManifest, shard_size: int = 4096) -> None:
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    rows: list[dict] = []; pending: list[dict] = []
    seen_keys: set[tuple[object, ...]] = set()

    def flush(shard_id: int, values: list[dict]) -> None:
        if not values:
            return
        name = f"phase_effect-{shard_id:05d}.safetensors"
        phase_pre = torch.stack([v["phase_pre"].float().cpu() for v in values])
        phase_post = torch.stack([v["phase_post"].float().cpu() for v in values])
        effect = torch.stack([v["effect"].float().cpu() for v in values])
        if phase_pre.ndim != 2 or phase_pre.shape[1] != manifest.phase_dim or phase_post.shape != phase_pre.shape:
            raise ValueError("phase feature shape does not match cache manifest")
        if effect.ndim != 2 or effect.shape[1] != manifest.effect_dim:
            raise ValueError("effect feature shape does not match cache manifest")
        if not (torch.isfinite(phase_pre).all() and torch.isfinite(phase_post).all() and torch.isfinite(effect).all()):
            raise ValueError("cache refuses non-finite phase/effect features")
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[manifest.feature_dtype]
        save_file({
            "phase_pre": phase_pre.to(dtype=dtype),
            "phase_post": phase_post.to(dtype=dtype),
            "effect": effect.to(dtype=dtype),
            "valid": torch.tensor([bool(v["valid"]) for v in values], dtype=torch.bool),
        }, str(root / name))
        for offset, value in enumerate(values):
            rows.append({"episode_id": str(value["episode_id"]), "task_id": value.get("task_id", 0),
                         "attempt_id": int(value.get("attempt_id", 0)),
                         "episode_step": None if value.get("episode_step") is None else int(value["episode_step"]),
                         "transition_index": int(value.get("transition_index", offset)),
                         **({"effect_index": int(value.get("effect_index", value.get("transition_index", offset)))} if manifest.schema_version in {"zeva_fastwam_robotwin_cache_v3", "zeva_fastwam_robotwin_cache_v4"} else {}),
                         **({"record_type": str(value.get("record_type", "effect"))} if manifest.schema_version == "zeva_fastwam_robotwin_cache_v4" else {}),
                         **({"raw_step": int(value["raw_step"])} if manifest.schema_version == "zeva_fastwam_robotwin_cache_v4" else {}),
                         **({"start_raw_step": int(value["start_raw_step"]), "end_raw_step": int(value["end_raw_step"])} if manifest.schema_version == "zeva_fastwam_robotwin_cache_v4" else {}),
                         "window_index": None if value.get("window_index") is None else int(value["window_index"]),
                         "shard": name, "offset": offset})

    shard_id = 0
    for record in records:
        if manifest.schema_version == "zeva_fastwam_robotwin_cache_v4":
            record_type = str(record.get("record_type", ""))
            if record_type not in {"phase_query", "effect"}:
                raise ValueError("v4 cache records must be phase_query or effect")
            for key in ("raw_step", "start_raw_step", "end_raw_step"):
                if key not in record:
                    raise ValueError(f"v4 cache record is missing {key}")
            _validate_v4_temporal_row(record, len(rows) + len(pending))
        if record.get("window_index") is not None:
            if manifest.schema_version == "zeva_fastwam_robotwin_cache_v4":
                key = (
                    int(record.get("window_index", -1)),
                    str(record.get("record_type", "effect")),
                    int(record.get("raw_step", record.get("episode_step", 0))),
                    int(record.get("effect_index", record.get("transition_index", 0))),
                )
            elif manifest.schema_version == "zeva_fastwam_robotwin_cache_v3":
                # One v3 row is one effect-window. The effect index is the
                # identity; transition_index is only a compatibility offset
                # and must not permit duplicate effect rows.
                key = (
                    int(record["window_index"]),
                    int(record.get("effect_index", record.get("transition_index", 0))),
                )
            else:
                key = (int(record["window_index"]), int(record.get("transition_index", 0)))
            if key in seen_keys:
                raise ValueError(f"duplicate cache row for window/effect {key}")
            seen_keys.add(key)
        pending.append(record)
        if len(pending) >= shard_size:
            flush(shard_id, pending); shard_id += 1; pending = []
    flush(shard_id, pending)
    (root / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    (root / "episode_index.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
