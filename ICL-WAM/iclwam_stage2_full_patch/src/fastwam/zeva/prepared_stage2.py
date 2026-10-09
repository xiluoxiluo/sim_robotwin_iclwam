"""Prepared offline inputs for scalable Zeva Stage-2 training.

The legacy :class:`ZevaStage2Dataset` computes PIM retrieval in Python for
 every training sample and reconstructs BIT prefixes from the full phase/effect
 cache at process start.  That is correct but unnecessarily expensive for
 RoboTwin-scale training.

This module defines an immutable, sharded mmap artifact containing exactly the
 tensors consumed by CausalPromptEncoder for every Stage-2 query window:

* current phase [D_phase]
* four causal BIT effect slots [B,D_effect] + mask
* top-K offline PIM phase/effect slots [K,D] + mask/scores

Static task context is stored separately because it is episode-level rather
 than query-window-level.  Both artifacts keep strong SHA256 identities so the
 prepared path cannot silently drift from the frozen CTE/base policy/bank/head.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from fastwam.datasets.zeva_robotwin_dataset import ZevaRobotWinDataset
from fastwam.zeva.checkpoint import checkpoint_sha256


PIM_CACHE_FORMAT = "zeva_stage2_prepared_pim_v1"
STATIC_CONTEXT_FORMAT = "zeva_static_episode_context_v1"


@dataclass(frozen=True)
class PreparedPIMShard:
    path: str
    global_start: int
    global_end: int

    @property
    def size(self) -> int:
        return self.global_end - self.global_start


class PreparedPIMCache:
    """Read a sharded mmap cache produced by build_zeva_pim_retrieval_cache.py."""

    _ARRAY_SPECS = {
        "window_index": np.int64,
        "raw_step": np.int32,
        "episode_index": np.int32,
        "task_index": np.int16,
        "phase": np.float32,
        "bit_effects": np.float32,
        "bit_mask": np.uint8,
        "pim_phases": np.float32,
        "pim_effects": np.float32,
        "pim_mask": np.uint8,
        "pim_scores": np.float32,
    }

    def __init__(self, root: str | Path, *, expected: dict[str, Any] | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"prepared PIM manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != PIM_CACHE_FORMAT:
            raise ValueError(f"unsupported prepared PIM format: {self.manifest.get('format')!r}")
        for key, value in dict(expected or {}).items():
            if self.manifest.get(key) != value:
                raise ValueError(
                    f"prepared PIM identity mismatch for {key}: "
                    f"expected={value!r}, actual={self.manifest.get(key)!r}"
                )

        self.phase_dim = int(self.manifest["phase_dim"])
        self.effect_dim = int(self.manifest["effect_dim"])
        self.top_k = int(self.manifest["top_k"])
        self.bit_size = int(self.manifest["bit_size"])
        self.num_queries = int(self.manifest["num_queries"])
        self.retrieval_mode = str(self.manifest["retrieval_mode"])
        if self.retrieval_mode != "phase":
            raise ValueError(
                "formal prepared Stage-2 cache currently supports only Zeva's "
                "default same-task phase retrieval"
            )
        if min(self.phase_dim, self.effect_dim, self.top_k, self.bit_size, self.num_queries) < 1:
            raise ValueError("prepared PIM manifest contains non-positive dimensions/counts")

        episode_ids = json.loads((self.root / "episode_ids.json").read_text(encoding="utf-8"))
        task_ids = json.loads((self.root / "task_ids.json").read_text(encoding="utf-8"))
        if not isinstance(episode_ids, list) or not episode_ids or len(set(episode_ids)) != len(episode_ids):
            raise ValueError("prepared PIM episode_ids must be a non-empty unique list")
        if not isinstance(task_ids, list) or not task_ids or len(set(task_ids)) != len(task_ids):
            raise ValueError("prepared PIM task_ids must be a non-empty unique list")
        self.episode_ids = tuple(str(v) for v in episode_ids)
        self.task_ids = tuple(str(v) for v in task_ids)
        if int(self.manifest.get("num_episodes", len(self.episode_ids))) != len(self.episode_ids):
            raise ValueError("prepared PIM episode_ids count differs from manifest")
        if int(self.manifest.get("semantic_task_count", len(self.task_ids))) != len(self.task_ids):
            raise ValueError("prepared PIM task_ids count differs from manifest")
        if len(self.task_ids) != 50:
            raise ValueError(f"formal RoboTwin prepared PIM requires 50 semantic tasks, got {len(self.task_ids)}")

        raw_shards = list(self.manifest.get("shards", []))
        self.shards = tuple(
            PreparedPIMShard(
                path=str(row["path"]),
                global_start=int(row["global_start"]),
                global_end=int(row["global_end"]),
            )
            for row in raw_shards
        )
        if not self.shards:
            raise ValueError("prepared PIM cache has no shards")
        cursor = 0
        for shard in self.shards:
            if shard.global_start != cursor or shard.global_end <= shard.global_start:
                raise ValueError("prepared PIM shards are not contiguous/non-empty")
            cursor = shard.global_end
        if cursor != self.num_queries:
            raise ValueError(
                f"prepared PIM shard coverage mismatch: shards={cursor}, manifest={self.num_queries}"
            )
        self._ends = [shard.global_end for shard in self.shards]
        self._opened: dict[int, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return self.num_queries

    def _open_shard(self, shard_index: int) -> dict[str, np.ndarray]:
        cached = self._opened.get(shard_index)
        if cached is not None:
            return cached
        shard = self.shards[shard_index]
        root = self.root / shard.path
        arrays: dict[str, np.ndarray] = {}
        for name, dtype in self._ARRAY_SPECS.items():
            path = root / f"{name}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"prepared PIM shard missing {path}")
            value = np.load(path, mmap_mode="r")
            if value.dtype != dtype:
                raise ValueError(
                    f"prepared PIM {path.name} dtype mismatch: {value.dtype} != {dtype}"
                )
            arrays[name] = value
        n = shard.size
        expected_shapes = {
            "window_index": (n,),
            "raw_step": (n,),
            "episode_index": (n,),
            "task_index": (n,),
            "phase": (n, self.phase_dim),
            "bit_effects": (n, self.bit_size, self.effect_dim),
            "bit_mask": (n, self.bit_size),
            "pim_phases": (n, self.top_k, self.phase_dim),
            "pim_effects": (n, self.top_k, self.effect_dim),
            "pim_mask": (n, self.top_k),
            "pim_scores": (n, self.top_k),
        }
        for name, shape in expected_shapes.items():
            if tuple(arrays[name].shape) != shape:
                raise ValueError(
                    f"prepared PIM {shard.path}/{name}.npy shape mismatch: "
                    f"{arrays[name].shape} != {shape}"
                )
        self._opened[shard_index] = arrays
        return arrays

    def _locate(self, index: int) -> tuple[int, int]:
        index = int(index)
        if index < 0:
            index += self.num_queries
        if index < 0 or index >= self.num_queries:
            raise IndexError(index)
        shard_index = bisect.bisect_right(self._ends, index)
        shard = self.shards[shard_index]
        return shard_index, index - shard.global_start

    @staticmethod
    def _tensor(value: np.ndarray, *, bool_value: bool = False) -> Tensor:
        array = np.array(value, copy=True)
        tensor = torch.from_numpy(array)
        return tensor.bool() if bool_value else tensor

    def get(self, index: int) -> dict[str, Any]:
        shard_index, local = self._locate(index)
        a = self._open_shard(shard_index)
        episode_index = int(a["episode_index"][local])
        task_index = int(a["task_index"][local])
        if not 0 <= episode_index < len(self.episode_ids):
            raise ValueError(f"prepared PIM episode_index out of range: {episode_index}")
        if not 0 <= task_index < len(self.task_ids):
            raise ValueError(f"prepared PIM task_index out of range: {task_index}")
        return {
            "window_index": int(a["window_index"][local]),
            "raw_step": int(a["raw_step"][local]),
            "episode_id": self.episode_ids[episode_index],
            "task_id": self.task_ids[task_index],
            "phase": self._tensor(a["phase"][local]).float(),
            "bit_effects": self._tensor(a["bit_effects"][local]).float(),
            "bit_mask": self._tensor(a["bit_mask"][local], bool_value=True),
            "pim_phases": self._tensor(a["pim_phases"][local]).float(),
            "pim_effects": self._tensor(a["pim_effects"][local]).float(),
            "pim_mask": self._tensor(a["pim_mask"][local], bool_value=True),
            "pim_scores": self._tensor(a["pim_scores"][local]).float(),
        }


class StaticEpisodeContextCache:
    """Episode-level static task-context values used only during Stage-2 training."""

    def __init__(self, path: str | Path, *, expected: dict[str, Any] | None = None) -> None:
        self.path = Path(path).expanduser().resolve()
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        if payload.get("format") != STATIC_CONTEXT_FORMAT:
            raise ValueError(f"unsupported static episode-context format: {payload.get('format')!r}")
        for key, value in dict(expected or {}).items():
            if payload.get(key) != value:
                raise ValueError(
                    f"static episode-context identity mismatch for {key}: "
                    f"expected={value!r}, actual={payload.get(key)!r}"
                )
        episode_ids = payload.get("episode_ids")
        values = payload.get("task_contexts")
        if not isinstance(episode_ids, list) or len(set(episode_ids)) != len(episode_ids):
            raise ValueError("static episode-context cache requires unique episode_ids")
        if not torch.is_tensor(values) or values.ndim != 2 or values.shape[0] != len(episode_ids):
            raise ValueError("static episode-context cache requires task_contexts [N,D]")
        if values.dtype != torch.float32 or not bool(torch.isfinite(values).all()):
            raise ValueError("static episode-context values must be finite float32")
        self.episode_ids = tuple(str(v) for v in episode_ids)
        self.values = values.contiguous()
        self.value_dim = int(values.shape[1])
        self.metadata = dict(payload.get("metadata", {}))
        self._index = {episode_id: i for i, episode_id in enumerate(self.episode_ids)}

    def __len__(self) -> int:
        return len(self.episode_ids)

    def get(self, episode_id: str) -> Tensor:
        try:
            index = self._index[str(episode_id)]
        except KeyError as exc:
            raise KeyError(f"static context missing episode {episode_id}") from exc
        return self.values[index].clone()


class PreparedZevaStage2Dataset(Dataset):
    """Fast Stage-2 dataset with O(1) mmap lookup for BIT/PIM/static context.

    The underlying RobotVideoDataset remains authoritative for RGB/action/text
    FastWAM inputs.  Prepared artifacts supply only frozen Zeva conditioning,
    and every sample re-checks window/episode/task/raw-step identity before use.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        pim_cache: PreparedPIMCache,
        static_context: StaticEpisodeContextCache,
    ) -> None:
        self.base = ZevaRobotWinDataset(base_dataset)
        self.pim = pim_cache
        self.static_context = static_context
        if self.static_context.value_dim < 1:
            raise ValueError("static task-context dimension must be positive")
        missing = set(self.pim.episode_ids) - set(self.static_context.episode_ids)
        if missing:
            raise ValueError(
                f"static context cache is missing {len(missing)} PIM episodes; "
                f"examples={sorted(missing)[:5]}"
            )

    def __len__(self) -> int:
        return len(self.pim)

    def __getitem__(self, index: int) -> dict[str, Any]:
        prepared = self.pim.get(index)
        requested_index = int(prepared["window_index"])
        sample = self.base[requested_index]
        source_index = int(sample.get("dataset_index", requested_index))
        if source_index != requested_index:
            raise RuntimeError(
                f"prepared Stage-2 deterministic-index mismatch: requested={requested_index}, got={source_index}"
            )
        episode = sample["episode"]
        if str(episode.episode_id) != str(prepared["episode_id"]):
            raise ValueError(
                f"prepared Stage-2 episode mismatch at window {source_index}: "
                f"sample={episode.episode_id}, cache={prepared['episode_id']}"
            )
        if str(episode.task_id) != str(prepared["task_id"]):
            raise ValueError(
                f"prepared Stage-2 semantic task mismatch at window {source_index}: "
                f"sample={episode.task_id}, cache={prepared['task_id']}"
            )
        if int(episode.episode_step) != int(prepared["raw_step"]):
            raise ValueError(
                f"prepared Stage-2 raw-step mismatch at window {source_index}: "
                f"sample={episode.episode_step}, cache={prepared['raw_step']}"
            )
        sample.update(
            {
                "phase": prepared["phase"],
                "bit_effects": prepared["bit_effects"],
                "bit_mask": prepared["bit_mask"],
                "pim_phases": prepared["pim_phases"],
                "pim_effects": prepared["pim_effects"],
                "pim_mask": prepared["pim_mask"],
                "pim_scores": prepared["pim_scores"],
                "raw_step": int(prepared["raw_step"]),
                "task_context": self.static_context.get(prepared["episode_id"]),
            }
        )
        sample.pop("episode", None)
        sample.pop("behavior_memory", None)
        sample.pop("behavior_memory_mask", None)
        return sample


def validate_static_artifact_hashes(
    static_cache: StaticEpisodeContextCache,
    *,
    bank_path: str | Path,
    head_path: str | Path,
    readout_path: str | Path,
) -> None:
    expected = {
        "bank_sha256": checkpoint_sha256(bank_path),
        "retrieval_checkpoint_sha256": checkpoint_sha256(head_path),
        "readout_cache_sha256": checkpoint_sha256(readout_path),
    }
    for key, value in expected.items():
        if static_cache.metadata.get(key) != value:
            raise ValueError(
                f"static context metadata mismatch for {key}: "
                f"expected={value}, actual={static_cache.metadata.get(key)}"
            )
