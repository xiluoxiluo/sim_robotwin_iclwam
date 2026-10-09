"""Build Zeva phase/effect cache from a frozen Stage-1 CTE.

Optimized RoboTwin implementation:

1. Build the cache query plan from LeRobot metadata only.  No RGB/video is
   decoded while indexing the ~6M source rows.
2. Keep only compact per-episode plans in memory.  Samples are decoded lazily,
   one episode at a time.
3. Decode only a minimal set of 32-action source windows that reconstructs the
   exact same full episode prefix required by cache-v4.  We do NOT decode every
   overlapping 4-step query window.
4. Support torchrun multi-GPU execution.  Episodes are greedily balanced across
   ranks by query count.
5. Use a bounded CPU ThreadPool prefetch pipeline so video/PyAV decoding of
   upcoming episodes overlaps VAE/CTE execution on the current PPU.
6. Feed decoded RGB to the VAE on-device, keep VAE latents on-device for CTE,
   and copy each CTE output tensor to CPU only once.
7. Stream CPU cache records into safetensors shards on an asynchronous writer
   thread instead of stalling the PPU on shard I/O.
8. Show metadata and per-rank cache progress bars with queue-wait diagnostics.
9. Each rank writes a private temporary cache; rank 0 merges the indices/shards
   into the final cache atomically after all ranks finish.

Single GPU:
    python scripts/build_zeva_robotwin_cache.py ...

Two GPUs:
    torchrun --standalone --nproc_per_node=2 \\
        scripts/build_zeva_robotwin_cache.py ...
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import shutil
import threading
import time
from typing import Any, Iterable, Iterator

import hydra
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from safetensors.torch import save_file
from tqdm import tqdm

from fastwam.datasets.zeva_robotwin_dataset import ZevaRobotWinDataset
from fastwam.zeva import (
    CausalTransitionEncoder,
    CausalTransitionEncoderConfig,
    FastWAMCTELatentEncoder,
    load_frozen_wan_vae,
    validate_vae_metadata,
)
from fastwam.zeva.checkpoint import checkpoint_sha256, load_cte_checkpoint
from fastwam.zeva.schemas import CacheManifest, sha256_file


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _video_size_hw(cfg: DictConfig) -> tuple[int, int]:
    value = cfg.data.train.get("video_size")
    if value is None or len(value) != 2:
        raise ValueError("data.train.video_size must be [H, W] for Zeva cache construction")
    size = tuple(int(v) for v in value)
    if min(size) < 1:
        raise ValueError(f"data.train.video_size must be positive, got {size}")
    return size


def _task_id_from_metadata(inner_dataset: Any, value: Any) -> str | int:
    """Match the existing Zeva task-id convention without decoding video."""
    try:
        task_id: str | int = int(value)
    except (TypeError, ValueError):
        task_id = str(value)

    # RobotVideoDataset uses task_index when present.  Existing Zeva indexing
    # replaces the shared numeric sentinel 0 with the corresponding task text.
    if task_id in (0, "0"):
        try:
            task_id = str(inner_dataset.meta.tasks[int(task_id)])
        except (KeyError, TypeError, ValueError, IndexError):
            task_id = f"task-{task_id}"
    return task_id


def _episode_value(episode: Any, name: str, default: Any = None) -> Any:
    if isinstance(episode, dict):
        return episode.get(name, default)
    return getattr(episode, name, default)



def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not str(value).strip():
        return int(default)
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _configure_cpu_runtime(world_size: int) -> int:
    """Bound per-rank torch CPU teams so decode workers do not oversubscribe."""
    threads = _env_int("ZEVA_CACHE_TORCH_CPU_THREADS", 2)
    threads = max(1, threads)
    torch.set_num_threads(threads)
    # set_num_interop_threads can only be called before inter-op work starts.
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return threads


def _configure_video_backend(cfg: DictConfig) -> str:
    """Avoid repeatedly trying a TorchCodec backend known to fail on this host."""
    env_backend = os.environ.get("ZEVA_CACHE_VIDEO_BACKEND", "").strip()
    configured = cfg.data.train.get("video_backend")
    if env_backend:
        backend = env_backend
    elif configured not in (None, "", "None", "null"):
        backend = str(configured)
    else:
        # The current PPU host reports TorchCodec 'Function not implemented'.
        # Going straight to PyAV avoids one failed decoder construction per read.
        backend = "pyav"
    with open_dict(cfg.data.train):
        cfg.data.train.video_backend = backend
    return backend


# -----------------------------------------------------------------------------
# Distributed setup
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DistInfo:
    rank: int
    local_rank: int
    world_size: int
    distributed: bool


def _init_distributed() -> DistInfo:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("multi-rank Zeva cache construction requires CUDA/NCCL")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")

    return DistInfo(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        distributed=distributed,
    )


def _barrier(info: DistInfo) -> None:
    if info.distributed:
        dist.barrier()


def _destroy_process_group(info: DistInfo) -> None:
    if info.distributed and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _resolve_device(cfg: DictConfig, info: DistInfo) -> torch.device:
    if info.distributed:
        return torch.device(f"cuda:{info.local_rank}")

    configured_device = cfg.get("device")
    if configured_device is None or str(configured_device).strip().lower() in {
        "",
        "none",
        "null",
    }:
        configured_device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(str(configured_device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("latent CTE cache requested CUDA but CUDA is unavailable")
    return device


# -----------------------------------------------------------------------------
# Metadata-only episode/query planning
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodePlan:
    episode_id: str
    task_id: str | int
    dataset_start: int
    episode_length: int
    max_query_start: int
    query_stride: int
    source_window_actions: int

    @property
    def query_count(self) -> int:
        return self.max_query_start // self.query_stride + 1

    def iter_query_starts(self) -> range:
        return range(0, self.max_query_start + 1, self.query_stride)

    def coverage_starts(self) -> list[int]:
        """Minimal source windows that reconstruct the old full-prefix union.

        Every decoded Zeva source sample covers 32 raw actions and the 9 visual
        boundaries at offsets 0,4,...,32.  Query positions are every 4 actions.

        Decoding source windows at 0,32,64,... reconstructs a contiguous prefix.
        Appending the last query start (when it is not already selected) extends
        that prefix to exactly ``last_query + 32`` -- the same final boundary the
        previous all-overlapping-window implementation produced.
        """
        starts = list(
            range(
                0,
                self.max_query_start + 1,
                self.source_window_actions,
            )
        )
        if not starts:
            starts = [0]
        if starts[-1] != self.max_query_start:
            starts.append(self.max_query_start)
        return starts


def _get_inner_lerobot_datasets(dataset: ZevaRobotWinDataset) -> list[Any]:
    zeva_base = getattr(dataset, "base_dataset", None)
    lerobot_base = (
        getattr(zeva_base, "lerobot_dataset", None)
        if zeva_base is not None
        else None
    )
    multi_dataset = (
        getattr(lerobot_base, "multi_dataset", None)
        if lerobot_base is not None
        else None
    )
    inner_datasets = (
        getattr(multi_dataset, "_datasets", None)
        if multi_dataset is not None
        else None
    )

    if inner_datasets is None:
        raise RuntimeError(
            "RoboTwin cache metadata fast path could not reach "
            "ZevaRobotWinDataset -> RobotVideoDataset -> BaseLerobotDataset -> "
            "MultiLeRobotDataset._datasets. Refusing to silently fall back to "
            "dataset[index] over millions of rows."
        )
    return list(inner_datasets)


def _build_episode_plans_metadata_only(
    dataset: ZevaRobotWinDataset,
    *,
    sample_stride: int,
    transition_steps: int,
    source_window_actions: int = 32,
    show_progress: bool = True,
) -> list[EpisodePlan]:
    """Build every valid cache-v4 query position without decoding RGB/video."""
    if sample_stride != 1:
        raise ValueError("RoboTwin Zeva V1 metadata cache planner requires sample_stride=1")
    if transition_steps < 1 or source_window_actions < 1:
        raise ValueError("transition_steps/source_window_actions must be positive")
    if source_window_actions % transition_steps != 0:
        raise ValueError("source_window_actions must be divisible by transition_steps")

    query_stride = transition_steps * sample_stride
    plans: list[EpisodePlan] = []
    global_dataset_offset = 0

    robot_video_dataset = dataset.base_dataset

    inner_datasets = _get_inner_lerobot_datasets(dataset)

    for inner_pos, inner_dataset in enumerate(inner_datasets):
        episode_data_index = inner_dataset.episode_data_index
        episode_from = episode_data_index["from"]
        episode_to = episode_data_index["to"]

        if inner_dataset.episodes is None:
            episode_ids = list(range(inner_dataset.meta.total_episodes))
        else:
            episode_ids = list(inner_dataset.episodes)

        if len(episode_ids) != len(episode_from):
            raise RuntimeError(
                "LeRobot episode metadata mismatch: "
                f"{len(episode_ids)=}, {len(episode_from)=}"
            )

        # One task_index lookup per episode is enough for RoboTwin: task identity
        # is episode-level metadata.  This avoids gathering task_index for every
        # overlapping 4-step query window.
        episode_starts = [int(episode_from[i]) for i in range(len(episode_ids))]
        raw_dataset = inner_dataset.hf_dataset.with_format(None)
        if "task_index" in raw_dataset.column_names and episode_starts:
            task_values = raw_dataset[episode_starts]["task_index"]
        else:
            task_values = [0] * len(episode_ids)

        if len(task_values) != len(episode_ids):
            raise RuntimeError("task_index metadata count does not match episode count")

        dataset_root = str(inner_dataset.root)
        iterator: Iterable[int] = range(len(episode_ids))
        iterator = tqdm(
            iterator,
            total=len(episode_ids),
            desc=f"Metadata query plan [{inner_pos + 1}/{len(inner_datasets)}]",
            dynamic_ncols=True,
            disable=not show_progress,
            unit="episode",
        )

        for episode_pos in iterator:
            episode_id = int(episode_ids[episode_pos])
            ep_start = int(episode_from[episode_pos])
            ep_end = int(episode_to[episode_pos])
            episode_length = ep_end - ep_start

            # A Zeva source sample uses 32 actions + 33 raw observations.
            max_start = episode_length - (source_window_actions + 1)
            if max_start < 0:
                continue

            # Cache queries must coincide with grouped CTE boundaries: 0,4,8,...
            max_query_start = (max_start // query_stride) * query_stride
            
            task_id = robot_video_dataset.resolve_semantic_task_id(
                episode_index=episode_id,
                raw_task_index=task_values[episode_pos],
                strict=True,
            )

            plans.append(
                EpisodePlan(
                    episode_id=f"{dataset_root}::episode-{episode_id}",
                    task_id=task_id,
                    dataset_start=global_dataset_offset + ep_start,
                    episode_length=episode_length,
                    max_query_start=max_query_start,
                    query_stride=query_stride,
                    source_window_actions=source_window_actions,
                )
            )

        global_dataset_offset += int(inner_dataset.num_frames)

    return plans


def _partition_episode_plans(
    plans: list[EpisodePlan],
    world_size: int,
) -> tuple[list[list[EpisodePlan]], list[int]]:
    """Greedy LPT-style balancing by phase-query count."""
    buckets: list[list[EpisodePlan]] = [[] for _ in range(world_size)]
    loads = [0 for _ in range(world_size)]

    for plan in sorted(plans, key=lambda p: p.query_count, reverse=True):
        target = min(range(world_size), key=lambda r: loads[r])
        buckets[target].append(plan)
        loads[target] += plan.query_count

    for bucket in buckets:
        bucket.sort(key=lambda p: p.episode_id)

    return buckets, loads


# -----------------------------------------------------------------------------
# Streaming safetensors writer
# -----------------------------------------------------------------------------


class StreamingCacheWriter:
    """Write cache tensors in bounded-memory shards while accumulating metadata."""

    def __init__(
        self,
        root: str | Path,
        manifest: CacheManifest,
        *,
        shard_size: int = 4096,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        self.shard_size = int(shard_size)
        if self.shard_size < 1:
            raise ValueError("shard_size must be positive")

        self._pending: list[dict[str, Any]] = []
        self.rows: list[dict[str, Any]] = []
        self.shard_id = 0
        self.num_records = 0

    def add(self, record: dict[str, Any]) -> None:
        self._validate_record(record)
        self._pending.append(record)
        self.num_records += 1
        if len(self._pending) >= self.shard_size:
            self.flush()

    def extend(self, records: Iterable[dict[str, Any]]) -> None:
        for record in records:
            self.add(record)

    def _validate_record(self, record: dict[str, Any]) -> None:
        record_type = str(record.get("record_type", ""))
        if record_type not in {"phase_query", "effect"}:
            raise ValueError(f"unsupported v4 record_type={record_type!r}")

        for key in ("episode_step", "raw_step", "start_raw_step", "end_raw_step"):
            if key not in record:
                raise ValueError(f"cache record missing {key}")

        episode_step = int(record["episode_step"])
        raw_step = int(record["raw_step"])
        start_raw_step = int(record["start_raw_step"])
        end_raw_step = int(record["end_raw_step"])

        if min(episode_step, raw_step, start_raw_step, end_raw_step) < 0:
            raise ValueError("cache temporal coordinates must be non-negative")
        if start_raw_step > end_raw_step:
            raise ValueError("cache start_raw_step exceeds end_raw_step")

        if record_type == "phase_query":
            if record.get("window_index") is None:
                raise ValueError("phase_query requires window_index")
            if not (episode_step == raw_step == start_raw_step == end_raw_step):
                raise ValueError("phase_query must identify one raw boundary")
        else:
            if record.get("window_index") is not None:
                raise ValueError("effect record must not have window_index")
            if start_raw_step == end_raw_step:
                raise ValueError("effect record must cover a positive interval")
            if episode_step != start_raw_step or raw_step != end_raw_step:
                raise ValueError("effect record temporal metadata is inconsistent")

    def flush(self) -> None:
        if not self._pending:
            return

        values = self._pending
        name = f"phase_effect-{self.shard_id:05d}.safetensors"
        self.shard_id += 1

        phase_pre = torch.stack([v["phase_pre"].float().cpu() for v in values])
        phase_post = torch.stack([v["phase_post"].float().cpu() for v in values])
        effect = torch.stack([v["effect"].float().cpu() for v in values])

        if (
            phase_pre.ndim != 2
            or phase_pre.shape[1] != self.manifest.phase_dim
            or phase_post.shape != phase_pre.shape
        ):
            raise ValueError("phase feature shape does not match cache manifest")
        if effect.ndim != 2 or effect.shape[1] != self.manifest.effect_dim:
            raise ValueError("effect feature shape does not match cache manifest")
        if not (
            torch.isfinite(phase_pre).all()
            and torch.isfinite(phase_post).all()
            and torch.isfinite(effect).all()
        ):
            raise ValueError("cache refuses non-finite phase/effect features")

        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.manifest.feature_dtype]

        save_file(
            {
                "phase_pre": phase_pre.to(dtype=dtype),
                "phase_post": phase_post.to(dtype=dtype),
                "effect": effect.to(dtype=dtype),
                "valid": torch.tensor(
                    [bool(v["valid"]) for v in values],
                    dtype=torch.bool,
                ),
            },
            str(self.root / name),
        )

        for offset, value in enumerate(values):
            self.rows.append(
                {
                    "episode_id": str(value["episode_id"]),
                    "task_id": value.get("task_id", 0),
                    "attempt_id": int(value.get("attempt_id", 0)),
                    "episode_step": int(value["episode_step"]),
                    "transition_index": int(value.get("transition_index", offset)),
                    "effect_index": int(
                        value.get(
                            "effect_index",
                            value.get("transition_index", offset),
                        )
                    ),
                    "record_type": str(value["record_type"]),
                    "raw_step": int(value["raw_step"]),
                    "start_raw_step": int(value["start_raw_step"]),
                    "end_raw_step": int(value["end_raw_step"]),
                    "window_index": (
                        None
                        if value.get("window_index") is None
                        else int(value["window_index"])
                    ),
                    "shard": name,
                    "offset": offset,
                }
            )

        self._pending = []

    def close(self) -> None:
        self.flush()
        (self.root / "manifest.json").write_text(
            json.dumps(self.manifest.to_dict(), indent=2, sort_keys=True) + "\n"
        )
        (self.root / "episode_index.json").write_text(
            json.dumps(self.rows, indent=2, sort_keys=True) + "\n"
        )


# -----------------------------------------------------------------------------
# Asynchronous writer
# -----------------------------------------------------------------------------


class AsyncCacheWriter:
    """Move safetensors/JSON shard work off the PPU consumer thread."""

    _STOP = object()

    def __init__(
        self,
        writer: StreamingCacheWriter,
        *,
        enabled: bool = True,
        queue_size: int = 4,
    ) -> None:
        self.writer = writer
        self.enabled = bool(enabled)
        self.queue_size = max(1, int(queue_size))
        self.submitted_records = 0
        self._error: BaseException | None = None
        self._queue: queue.Queue[Any] | None = None
        self._thread: threading.Thread | None = None

        if self.enabled:
            self._queue = queue.Queue(maxsize=self.queue_size)
            self._thread = threading.Thread(
                target=self._run,
                name="zeva-cache-writer",
                daemon=True,
            )
            self._thread.start()

    def _run(self) -> None:
        assert self._queue is not None
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is self._STOP:
                        return
                    self.writer.extend(item)
                finally:
                    self._queue.task_done()
        except BaseException as exc:  # propagate to the main thread
            self._error = exc

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("asynchronous cache writer failed") from self._error

    def submit(self, records: list[dict[str, Any]]) -> None:
        self._raise_if_failed()
        self.submitted_records += len(records)
        if not self.enabled:
            self.writer.extend(records)
            return

        assert self._queue is not None
        while True:
            self._raise_if_failed()
            try:
                self._queue.put(records, timeout=0.25)
                return
            except queue.Full:
                continue

    def close(self) -> None:
        if not self.enabled:
            self.writer.close()
            return

        assert self._queue is not None
        assert self._thread is not None
        while True:
            self._raise_if_failed()
            try:
                self._queue.put(self._STOP, timeout=0.25)
                break
            except queue.Full:
                continue
        self._thread.join()
        self._raise_if_failed()
        self.writer.close()


# -----------------------------------------------------------------------------
# Episode decode / producer-consumer pipeline
# -----------------------------------------------------------------------------


@dataclass
class DecodedEpisode:
    plan: EpisodePlan
    boundary_frames: dict[int, torch.Tensor]
    action_groups: dict[int, torch.Tensor]
    segments: list[list[int]]
    preencoded: bool
    num_decoded_windows: int
    decode_seconds: float


@dataclass
class EpisodeComputeStats:
    total_seconds: float = 0.0
    h2d_seconds: float = 0.0
    vae_seconds: float = 0.0
    cte_seconds: float = 0.0


def _source_frames(sample: dict[str, Any]) -> tuple[torch.Tensor, bool]:
    frames = sample.get("cte_frames")
    preencoded = frames is not None
    if frames is None:
        frames = torch.cat(
            (sample["before_frames"], sample["after_frames"][-1:]),
            dim=0,
        )
    if frames.ndim != 4 or frames.shape[0] != 9:
        raise ValueError("sample['cte_frames'] must be [9,C,H,W]")
    return frames, preencoded


def _insert_consistent(
    mapping: dict[int, torch.Tensor],
    raw_step: int,
    value: torch.Tensor,
    *,
    kind: str,
    episode_id: str,
) -> None:
    previous = mapping.get(raw_step)
    if previous is not None:
        if not torch.allclose(previous, value, atol=1.0e-6, rtol=0.0):
            raise ValueError(
                f"inconsistent {kind} at episode={episode_id} raw_step={raw_step}"
            )
        return
    # Clone only unique boundaries/actions.  Keeping a view would retain the
    # complete decoded 9-frame source sample and defeat bounded prefetch memory.
    mapping[raw_step] = value.detach().cpu().clone()


def _decode_episode(
    plan: EpisodePlan,
    *,
    dataset: ZevaRobotWinDataset,
    transition_steps: int,
    sample_stride: int,
    cte_input_type: str,
) -> DecodedEpisode:
    """CPU producer: decode the minimal source windows for one episode."""
    started = time.perf_counter()
    boundary_frames: dict[int, torch.Tensor] = {}
    action_groups: dict[int, torch.Tensor] = {}
    encoded_flags: list[bool] = []
    coverage_starts = plan.coverage_starts()

    for episode_step in coverage_starts:
        dataset_index = plan.dataset_start + episode_step
        sample = dataset[dataset_index]

        source_index = int(sample.get("dataset_index", dataset_index))
        if source_index != dataset_index:
            raise RuntimeError(
                f"cache requires deterministic source indexing: requested "
                f"{dataset_index}, got {source_index}"
            )

        episode = sample.get("episode")
        if episode is None:
            raise ValueError(f"sample {dataset_index} is missing episode metadata")
        actual_episode_id = str(_episode_value(episode, "episode_id", ""))
        actual_step = int(_episode_value(episode, "episode_step", -1))
        if actual_episode_id != plan.episode_id or actual_step != episode_step:
            raise ValueError(
                "metadata/decode mismatch: "
                f"plan=({plan.episode_id}, step={episode_step}) "
                f"sample=({actual_episode_id}, step={actual_step})"
            )

        frames, preencoded = _source_frames(sample)
        encoded_flags.append(preencoded)

        for local_index, frame in enumerate(frames):
            raw_step = episode_step + local_index * transition_steps * sample_stride
            _insert_consistent(
                boundary_frames,
                raw_step,
                frame,
                kind="RGB/latent boundary",
                episode_id=plan.episode_id,
            )

        for local_index, action_group in enumerate(sample["transition_actions"]):
            raw_step = episode_step + local_index * transition_steps * sample_stride
            _insert_consistent(
                action_groups,
                raw_step,
                action_group,
                kind="action group",
                episode_id=plan.episode_id,
            )

        del sample, frames

    if any(encoded_flags) and not all(encoded_flags):
        raise ValueError("a full-history episode cannot mix RGB and preencoded CTE frames")
    preencoded = bool(encoded_flags and all(encoded_flags))
    if cte_input_type == "rgb_frame" and preencoded:
        raise ValueError("rgb_frame CTE cache cannot consume preencoded latent frames")

    raw_steps = sorted(boundary_frames)
    segments: list[list[int]] = []
    segment: list[int] = []
    expected_delta = transition_steps * sample_stride
    for raw_step in raw_steps:
        if segment and (
            raw_step != segment[-1] + expected_delta
            or segment[-1] not in action_groups
        ):
            segments.append(segment)
            segment = []
        segment.append(raw_step)
    if segment:
        segments.append(segment)

    return DecodedEpisode(
        plan=plan,
        boundary_frames=boundary_frames,
        action_groups=action_groups,
        segments=segments,
        preencoded=preencoded,
        num_decoded_windows=len(coverage_starts),
        decode_seconds=time.perf_counter() - started,
    )


def _iter_prefetched_decodes(
    plans: list[EpisodePlan],
    *,
    dataset: ZevaRobotWinDataset,
    transition_steps: int,
    sample_stride: int,
    cte_input_type: str,
    decode_workers: int,
    prefetch_episodes: int,
) -> Iterator[tuple[DecodedEpisode, float]]:
    """Yield decoded episodes while CPU workers stay ahead of the PPU."""
    workers = max(1, int(decode_workers))
    depth = max(workers, int(prefetch_episodes))

    if workers == 1:
        for plan in plans:
            wait_started = time.perf_counter()
            decoded = _decode_episode(
                plan,
                dataset=dataset,
                transition_steps=transition_steps,
                sample_stride=sample_stride,
                cte_input_type=cte_input_type,
            )
            # With one worker, all decode time is PPU queue wait.
            yield decoded, time.perf_counter() - wait_started
        return

    plan_iter = iter(plans)
    executor = ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="zeva-decode",
    )
    pending: dict[Future[DecodedEpisode], EpisodePlan] = {}

    def submit_next() -> bool:
        try:
            plan = next(plan_iter)
        except StopIteration:
            return False
        future = executor.submit(
            _decode_episode,
            plan,
            dataset=dataset,
            transition_steps=transition_steps,
            sample_stride=sample_stride,
            cte_input_type=cte_input_type,
        )
        pending[future] = plan
        return True

    try:
        for _ in range(min(depth, len(plans))):
            if not submit_next():
                break

        while pending:
            wait_started = time.perf_counter()
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            queue_wait = time.perf_counter() - wait_started

            ready: list[DecodedEpisode] = []
            for future in done:
                pending.pop(future)
                ready.append(future.result())
                # Refill before yielding so workers decode while the PPU consumes.
                submit_next()

            ready.sort(key=lambda item: item.plan.episode_id)
            for index, decoded in enumerate(ready):
                yield decoded, queue_wait if index == 0 else 0.0
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def _device_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _compute_episode(
    decoded: DecodedEpisode,
    *,
    model: CausalTransitionEncoder,
    cte_device: torch.device,
    cte_input_type: str,
    frame_encoder: Any,
    profile_gpu_timing: bool,
    pin_memory: bool,
) -> tuple[list[dict[str, Any]], EpisodeComputeStats]:
    """PPU consumer: VAE + CTE, returning only small CPU feature vectors."""
    started = time.perf_counter()
    stats = EpisodeComputeStats()
    plan = decoded.plan
    pending_queries = set(plan.iter_query_starts())
    episode_records: list[dict[str, Any]] = []

    for segment in decoded.segments:
        if len(segment) < 2:
            continue

        frames_cpu = torch.stack(
            [decoded.boundary_frames[raw_step] for raw_step in segment]
        )
        actions_cpu = torch.stack(
            [decoded.action_groups[raw_step] for raw_step in segment[:-1]]
        )

        if pin_memory and cte_device.type == "cuda":
            frames_cpu = frames_cpu.pin_memory()
            actions_cpu = actions_cpu.pin_memory()

        h2d_started = time.perf_counter()
        frames_device = frames_cpu.unsqueeze(0).to(
            cte_device,
            non_blocking=bool(pin_memory and cte_device.type == "cuda"),
        )
        actions_device = actions_cpu.unsqueeze(0).to(
            cte_device,
            non_blocking=bool(pin_memory and cte_device.type == "cuda"),
        )
        if profile_gpu_timing:
            _device_sync(cte_device)
            stats.h2d_seconds += time.perf_counter() - h2d_started

        if (
            cte_input_type == "wan_vae_latent"
            and frame_encoder is not None
            and not decoded.preencoded
        ):
            # Important: feed GPU RGB directly to the adapter.  The previous
            # implementation encoded on the PPU, copied latents to CPU, then
            # copied them straight back to the PPU for CTE.
            vae_started = time.perf_counter()
            frames_device = frame_encoder(frames_device)
            if profile_gpu_timing:
                _device_sync(cte_device)
                stats.vae_seconds += time.perf_counter() - vae_started

        cte_started = time.perf_counter()
        output = model(
            frames_device,
            actions_device,
            valid_mask=torch.ones(
                (1, len(segment)),
                dtype=torch.bool,
                device=cte_device,
            ),
            transition_valid=torch.ones(
                (1, len(segment) - 1, actions_device.shape[2]),
                dtype=torch.bool,
                device=cte_device,
            ),
        )

        # One batched D2H copy per output tensor.  Do not call .cpu() once per
        # query/effect: that would repeatedly synchronize the PPU stream.
        phase_cpu = output["phase"][0].detach().float().cpu().contiguous()
        effect_post_cpu = (
            output["effect_post"][0].detach().float().cpu().contiguous()
        )
        effect_complete_cpu = (
            output["effect_complete"][0].detach().bool().cpu().contiguous()
        )
        if profile_gpu_timing:
            stats.cte_seconds += time.perf_counter() - cte_started

        raw_to_index = {raw_step: i for i, raw_step in enumerate(segment)}

        for query_raw_step in sorted(pending_queries & raw_to_index.keys()):
            query_index = raw_to_index[query_raw_step]
            query_phase = phase_cpu[query_index].clone()
            window_index = plan.dataset_start + query_raw_step
            episode_records.append(
                {
                    "record_type": "phase_query",
                    "episode_id": plan.episode_id,
                    "task_id": plan.task_id,
                    "attempt_id": 0,
                    "window_index": window_index,
                    "episode_step": query_raw_step,
                    "raw_step": query_raw_step,
                    "start_raw_step": query_raw_step,
                    "end_raw_step": query_raw_step,
                    "transition_index": query_index,
                    "effect_index": 0,
                    "phase_pre": query_phase,
                    "phase_post": query_phase.clone(),
                    "effect": torch.zeros(
                        model.cfg.effect_dim,
                        dtype=torch.float32,
                    ),
                    "valid": True,
                }
            )
            pending_queries.remove(query_raw_step)

        for effect_index in range(effect_post_cpu.shape[0]):
            if not bool(effect_complete_cpu[effect_index].item()):
                continue

            start_index = effect_index * model.cfg.effect_window_transitions
            end_index = start_index + model.cfg.effect_window_transitions
            if end_index >= len(segment):
                raise RuntimeError(
                    "CTE effect index exceeds reconstructed episode segment: "
                    f"episode={plan.episode_id}, effect_index={effect_index}, "
                    f"segment_len={len(segment)}"
                )

            start_raw_step = segment[start_index]
            end_raw_step = segment[end_index]
            episode_records.append(
                {
                    "record_type": "effect",
                    "episode_id": plan.episode_id,
                    "task_id": plan.task_id,
                    "attempt_id": 0,
                    "window_index": None,
                    "episode_step": start_raw_step,
                    "raw_step": end_raw_step,
                    "start_raw_step": start_raw_step,
                    "end_raw_step": end_raw_step,
                    "transition_index": start_index,
                    "effect_index": effect_index,
                    "phase_pre": phase_cpu[start_index].clone(),
                    "phase_post": phase_cpu[end_index].clone(),
                    "effect": effect_post_cpu[effect_index].clone(),
                    "valid": True,
                }
            )

        del (
            output,
            phase_cpu,
            effect_post_cpu,
            effect_complete_cpu,
            frames_device,
            actions_device,
            frames_cpu,
            actions_cpu,
        )

    if pending_queries:
        missing = sorted(pending_queries)
        raise RuntimeError(
            f"reconstructed episode prefix missed {len(missing)} phase queries "
            f"for {plan.episode_id}; first missing={missing[:8]}"
        )

    record_order = {"effect": 0, "phase_query": 1}
    episode_records.sort(
        key=lambda row: (
            int(row["raw_step"]),
            record_order[row["record_type"]],
            int(row.get("effect_index", 0)),
        )
    )
    stats.total_seconds = time.perf_counter() - started
    return episode_records, stats


# -----------------------------------------------------------------------------
# Rank-cache merge
# -----------------------------------------------------------------------------


def _merge_rank_caches(
    *,
    work_root: Path,
    final_root: Path,
    manifest: CacheManifest,
    world_size: int,
    overwrite: bool,
) -> int:
    merged_root = work_root / "merged"
    if merged_root.exists():
        shutil.rmtree(merged_root)
    merged_root.mkdir(parents=True, exist_ok=True)

    expected_manifest = manifest.to_dict()
    all_rows: list[dict[str, Any]] = []

    for rank in range(world_size):
        rank_root = work_root / f"rank_{rank:03d}"
        manifest_path = rank_root / "manifest.json"
        index_path = rank_root / "episode_index.json"
        if not manifest_path.is_file() or not index_path.is_file():
            raise FileNotFoundError(f"rank {rank} cache output is incomplete: {rank_root}")

        rank_manifest = CacheManifest(**json.loads(manifest_path.read_text()))
        rank_manifest.validate(expected_manifest)
        rows = json.loads(index_path.read_text())

        shard_map: dict[str, str] = {}
        for old_name in sorted({str(row["shard"]) for row in rows}):
            src = rank_root / old_name
            if not src.is_file():
                raise FileNotFoundError(f"rank cache references missing shard: {src}")
            new_name = f"rank{rank:03d}-{old_name}"
            dst = merged_root / new_name
            shutil.move(str(src), str(dst))
            shard_map[old_name] = new_name

        for row in rows:
            row["shard"] = shard_map[str(row["shard"])]
        all_rows.extend(rows)

    record_order = {"effect": 0, "phase_query": 1}
    all_rows.sort(
        key=lambda row: (
            str(row["episode_id"]),
            int(row["raw_step"]),
            record_order[str(row["record_type"])],
            int(row.get("effect_index", 0)),
            -1 if row.get("window_index") is None else int(row["window_index"]),
        )
    )

    # Cheap deterministic duplicate check for phase-query source indices.
    seen_windows: set[int] = set()
    for row in all_rows:
        if row.get("record_type") != "phase_query":
            continue
        window_index = int(row["window_index"])
        if window_index in seen_windows:
            raise ValueError(f"duplicate cache window_index {window_index}")
        seen_windows.add(window_index)

    (merged_root / "manifest.json").write_text(
        json.dumps(expected_manifest, indent=2, sort_keys=True) + "\n"
    )
    (merged_root / "episode_index.json").write_text(
        json.dumps(all_rows, indent=2, sort_keys=True) + "\n"
    )

    if final_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"cache output already exists: {final_root}. "
                "Set +model.zeva.cache.overwrite=true to replace it."
            )
        shutil.rmtree(final_root)

    final_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(merged_root), str(final_root))
    return len(all_rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    info = _init_distributed()
    torch_cpu_threads = _configure_cpu_runtime(info.world_size)
    video_backend = _configure_video_backend(cfg)

    try:
        zeva = cfg.model.get("zeva", {})
        cache_cfg = zeva.get("cache", {})
        cte_path = str(zeva.get("cte", {}).get("checkpoint"))
        cache_path = str(zeva.get("cache", {}).get("path"))
        if (
            not cte_path
            or cte_path in {"None", "null"}
            or not cache_path
            or cache_path in {"None", "null"}
        ):
            raise ValueError("Set model.zeva.cte.checkpoint and model.zeva.cache.path")

        stats_path = str(cfg.data.train.get("pretrained_norm_stats", ""))
        if stats_path in {"", "None", "null"} or not Path(stats_path).is_file():
            raise FileNotFoundError(
                "Zeva cache construction requires an existing "
                "data.train.pretrained_norm_stats file"
            )

        sample_stride = int(cfg.data.train.get("global_sample_stride", 1))
        if sample_stride != 1:
            raise ValueError(
                "RoboTwin Zeva V1 requires data.train.global_sample_stride=1 "
                f"for exact frame/action alignment; got {sample_stride}"
            )

        # ------------------------------------------------------------------
        # Load and validate Stage-1 CTE metadata.
        # ------------------------------------------------------------------
        payload = torch.load(cte_path, map_location="cpu", weights_only=False)
        cte_config = dict(payload.get("config", payload.get("model_config", {})))
        cte_config = dict(cte_config.get("cte", cte_config))
        cte_input_type = str(
            payload.get("cte_input_type", cte_config.get("input_type", "rgb_frame"))
        )
        checkpoint_action_dim = int(
            payload.get("action_dim", cte_config.get("action_dim", -1))
        )

        if (
            checkpoint_action_dim != 14
            or cte_input_type not in {"rgb_frame", "wan_vae_latent"}
            or tuple(payload.get("camera_keys", ()))
            != ("cam_high", "cam_left_wrist", "cam_right_wrist")
        ):
            raise ValueError(
                "CTE checkpoint metadata is incompatible with RoboTwin V1 "
                "(input_type=rgb_frame|wan_vae_latent, action_dim=14, "
                "cameras=cam_high/cam_left_wrist/cam_right_wrist)"
            )

        checkpoint_vae_metadata = dict(payload.get("vae_metadata", {}))
        cte_vae_input_size = _video_size_hw(cfg)
        checkpoint_input_size = payload.get("cte_vae_input_size")

        if cte_input_type == "wan_vae_latent":
            if checkpoint_input_size is None or len(checkpoint_input_size) != 2:
                raise ValueError(
                    "wan_vae_latent CTE checkpoints must include cte_vae_input_size"
                )
            checkpoint_input_size = tuple(int(v) for v in checkpoint_input_size)
            if checkpoint_input_size != cte_vae_input_size:
                raise ValueError(
                    "CTE/cache VAE input size mismatch: checkpoint declares "
                    f"{checkpoint_input_size}, data config uses {cte_vae_input_size}"
                )
            if not checkpoint_vae_metadata:
                raise ValueError("wan_vae_latent CTE checkpoints must include vae_metadata")

            required_vae_metadata = {
                "model_id",
                "vae_path",
                "z_dim",
                "temporal_downsample_factor",
                "upsampling_factor",
            }
            if not required_vae_metadata.issubset(checkpoint_vae_metadata):
                raise ValueError(
                    "wan_vae_latent CTE checkpoints must record complete VAE "
                    f"identity metadata: {sorted(required_vae_metadata)}"
                )

        allowed = set(CausalTransitionEncoderConfig.__dataclass_fields__)
        model = CausalTransitionEncoder(
            CausalTransitionEncoderConfig(
                **{k: v for k, v in cte_config.items() if k in allowed}
            )
        )
        if (
            model.cfg.action_dim != 14
            or model.cfg.transition_steps != 4
            or model.cfg.effect_window_transitions != 4
        ):
            raise ValueError(
                "RoboTwin Zeva V1 cache requires CTE action_dim=14, "
                "transition_steps=4, and effect_window_transitions=4; "
                f"got action_dim={model.cfg.action_dim}, "
                f"transition_steps={model.cfg.transition_steps}, "
                f"effect_window_transitions={model.cfg.effect_window_transitions}"
            )
        if int(cfg.data.train.action_video_freq_ratio) != 4:
            raise ValueError(
                "RoboTwin Zeva V1 requires action_video_freq_ratio=4; "
                f"got {cfg.data.train.action_video_freq_ratio}"
            )

        load_cte_checkpoint(cte_path, model, map_location="cpu")

        # ------------------------------------------------------------------
        # Build the base dataset.  With torchrun, rank 0 creates/warms the HF
        # cache first so ranks do not race while materializing the train split.
        # ------------------------------------------------------------------
        base = None
        if info.distributed:
            if info.rank == 0:
                base = instantiate(cfg.data.train)
            _barrier(info)
            if info.rank != 0:
                base = instantiate(cfg.data.train)
            _barrier(info)
        else:
            base = instantiate(cfg.data.train)

        assert base is not None
        dataset = ZevaRobotWinDataset(base)

        # Metadata-only planning is cheap enough for each rank to repeat and
        # avoids broadcasting a very large Python object graph.
        plans = _build_episode_plans_metadata_only(
            dataset,
            sample_stride=sample_stride,
            transition_steps=model.cfg.transition_steps,
            source_window_actions=model.cfg.transition_steps * 8,
            show_progress=(info.rank == 0),
        )
        if not plans:
            raise RuntimeError("cache metadata planner found no complete RoboTwin episodes")

        assignments, loads = _partition_episode_plans(plans, info.world_size)
        local_plans = assignments[info.rank]

        if info.rank == 0:
            total_queries = sum(plan.query_count for plan in plans)
            print(
                f"[cache plan] episodes={len(plans):,} "
                f"phase_queries={total_queries:,} world_size={info.world_size}"
            )
            for rank, (rank_plans, load) in enumerate(zip(assignments, loads, strict=True)):
                print(
                    f"[cache plan] rank={rank}: episodes={len(rank_plans):,}, "
                    f"phase_queries={load:,}"
                )

        # ------------------------------------------------------------------
        # CPU decode / writer pipeline settings.
        # ------------------------------------------------------------------
        logical_cpus = max(1, os.cpu_count() or 1)
        cpus_per_rank = max(1, logical_cpus // info.world_size)
        auto_decode_workers = min(8, max(2, cpus_per_rank // 4))
        decode_workers = int(
            cache_cfg.get(
                "decode_workers",
                _env_int("ZEVA_CACHE_DECODE_WORKERS", 0),
            )
        )
        if decode_workers <= 0:
            decode_workers = auto_decode_workers
        prefetch_episodes = int(
            cache_cfg.get(
                "prefetch_episodes",
                _env_int("ZEVA_CACHE_PREFETCH_EPISODES", 0),
            )
        )
        if prefetch_episodes <= 0:
            prefetch_episodes = max(decode_workers * 2, decode_workers)
        async_write = bool(
            cache_cfg.get(
                "async_write",
                _env_bool("ZEVA_CACHE_ASYNC_WRITE", True),
            )
        )
        writer_queue_size = int(
            cache_cfg.get(
                "writer_queue_size",
                _env_int("ZEVA_CACHE_WRITER_QUEUE", 4),
            )
        )
        pin_memory = bool(
            cache_cfg.get(
                "pin_memory",
                _env_bool("ZEVA_CACHE_PIN_MEMORY", False),
            )
        )
        profile_gpu_timing = bool(
            cache_cfg.get(
                "profile_gpu_timing",
                _env_bool("ZEVA_CACHE_PROFILE_GPU", False),
            )
        )

        if info.rank == 0:
            print(
                "[cache runtime] "
                f"video_backend={video_backend} "
                f"logical_cpus={logical_cpus} "
                f"torch_cpu_threads/rank={torch_cpu_threads} "
                f"decode_workers/rank={decode_workers} "
                f"prefetch_episodes/rank={prefetch_episodes} "
                f"async_write={async_write} "
                f"pin_memory={pin_memory}"
            )

        # ------------------------------------------------------------------
        # Per-rank device / frozen VAE / frozen CTE.
        # ------------------------------------------------------------------
        cte_device = _resolve_device(cfg, info)
        frame_encoder = None
        vae_metadata: dict[str, object] = {}

        if cte_input_type == "wan_vae_latent":
            model_values = dict(cfg.model)
            vae, vae_metadata = load_frozen_wan_vae(
                model_id=str(
                    model_values.get("model_id", "Wan-AI/Wan2.2-TI2V-5B")
                ),
                tokenizer_model_id=str(
                    model_values.get(
                        "tokenizer_model_id",
                        "Wan-AI/Wan2.1-T2V-1.3B",
                    )
                ),
                device=str(cte_device),
                torch_dtype=(
                    torch.float32 if cte_device.type == "cpu" else torch.bfloat16
                ),
                redirect_common_files=bool(
                    model_values.get("redirect_common_files", True)
                ),
            )
            validate_vae_metadata(checkpoint_vae_metadata, vae_metadata)
            frame_encoder = FastWAMCTELatentEncoder(
                vae,
                resize=cte_vae_input_size,
                expected_channels=model.cfg.image_channels,
                input_range="minus_one_one",
            ).encode_history

        model.to(cte_device)
        model.eval().requires_grad_(False)

        manifest = CacheManifest(
            schema_version="zeva_fastwam_robotwin_cache_v4",
            history_semantics="full_episode_prefix",
            query_step_unit="raw_action_step",
            cte_checkpoint_sha256=checkpoint_sha256(cte_path),
            dataset_stats_sha256=sha256_file(stats_path),
            dataset_path=str(cfg.data.train.dataset_dirs[0]),
            action_dim=model.cfg.action_dim,
            action_group_size=model.cfg.transition_steps,
            action_horizon=model.cfg.transition_steps * 8,
            video_frames=9,
            phase_dim=model.cfg.phase_dim,
            effect_dim=model.cfg.effect_dim,
            image_channels=model.cfg.image_channels,
            feature_dtype=str(cache_cfg.get("feature_dtype", "float32")),
            action_video_freq_ratio=int(cfg.data.train.action_video_freq_ratio),
            cte_input_type=cte_input_type,
            latent_channels=(
                model.cfg.image_channels if cte_input_type == "wan_vae_latent" else 0
            ),
            vae_metadata=dict(vae_metadata),
            cte_vae_input_size=(
                cte_vae_input_size if cte_input_type == "wan_vae_latent" else None
            ),
        )

        # Every rank should see identical VAE identity metadata.  CacheManifest
        # validation during merge will catch any accidental mismatch.
        final_root = Path(cache_path)
        work_root = final_root.parent / f".{final_root.name}.building"
        rank_root = work_root / f"rank_{info.rank:03d}"

        if info.rank == 0:
            if work_root.exists():
                shutil.rmtree(work_root)
            work_root.mkdir(parents=True, exist_ok=True)
        _barrier(info)
        rank_root.mkdir(parents=True, exist_ok=True)

        shard_size = int(cache_cfg.get("shard_size", 4096))
        writer = StreamingCacheWriter(
            rank_root,
            manifest,
            shard_size=shard_size,
        )
        async_writer = AsyncCacheWriter(
            writer,
            enabled=async_write,
            queue_size=writer_queue_size,
        )

        local_query_total = sum(plan.query_count for plan in local_plans)
        progress = tqdm(
            total=local_query_total,
            desc=f"Zeva cache GPU{info.local_rank} rank {info.rank}/{info.world_size}",
            unit="query",
            dynamic_ncols=True,
            position=info.rank,
            leave=True,
            disable=not bool(cache_cfg.get("show_progress", True)),
        )

        decoded_windows = 0
        processed_episodes = 0
        queue_wait_seconds = 0.0
        decode_worker_seconds = 0.0
        compute_seconds = 0.0
        h2d_seconds = 0.0
        vae_seconds = 0.0
        cte_seconds = 0.0
        run_started = time.perf_counter()

        try:
            with torch.inference_mode():
                for decoded, queue_wait in _iter_prefetched_decodes(
                    local_plans,
                    dataset=dataset,
                    transition_steps=model.cfg.transition_steps,
                    sample_stride=sample_stride,
                    cte_input_type=cte_input_type,
                    decode_workers=decode_workers,
                    prefetch_episodes=prefetch_episodes,
                ):
                    queue_wait_seconds += queue_wait
                    decode_worker_seconds += decoded.decode_seconds

                    episode_records, compute_stats = _compute_episode(
                        decoded,
                        model=model,
                        cte_device=cte_device,
                        cte_input_type=cte_input_type,
                        frame_encoder=frame_encoder,
                        profile_gpu_timing=profile_gpu_timing,
                        pin_memory=pin_memory,
                    )
                    async_writer.submit(episode_records)

                    processed_episodes += 1
                    decoded_windows += decoded.num_decoded_windows
                    compute_seconds += compute_stats.total_seconds
                    h2d_seconds += compute_stats.h2d_seconds
                    vae_seconds += compute_stats.vae_seconds
                    cte_seconds += compute_stats.cte_seconds

                    progress.update(decoded.plan.query_count)
                    elapsed = max(1.0e-9, time.perf_counter() - run_started)
                    progress.set_postfix(
                        ep=f"{processed_episodes}/{len(local_plans)}",
                        decW=decoded_windows,
                        rec=async_writer.submitted_records,
                        wait=f"{100.0 * queue_wait_seconds / elapsed:.1f}%",
                        cmp=f"{compute_seconds / max(1, processed_episodes):.2f}s/ep",
                    )

                    # Release large decoded episode buffers as soon as PPU
                    # consumption has completed.  The next episodes are already
                    # being decoded by producer threads.
                    del decoded, episode_records
        finally:
            progress.close()
            async_writer.close()

        run_seconds = time.perf_counter() - run_started
        print(
            f"[rank {info.rank}] finished: episodes={len(local_plans):,}, "
            f"decoded_windows={decoded_windows:,}, "
            f"records={async_writer.submitted_records:,}, "
            f"wall={run_seconds:.1f}s, "
            f"queue_wait={queue_wait_seconds:.1f}s "
            f"({100.0 * queue_wait_seconds / max(run_seconds, 1e-9):.1f}%), "
            f"decode_worker_sum={decode_worker_seconds:.1f}s, "
            f"compute={compute_seconds:.1f}s, output={rank_root}"
        )
        if profile_gpu_timing:
            print(
                f"[rank {info.rank}] timing: h2d={h2d_seconds:.1f}s "
                f"vae={vae_seconds:.1f}s cte+d2h={cte_seconds:.1f}s"
            )

        _barrier(info)

        # ------------------------------------------------------------------
        # Rank 0 merges rank-local shard metadata.  Tensors are never gathered
        # through NCCL and never reloaded into GPU memory.
        # ------------------------------------------------------------------
        if info.rank == 0:
            overwrite = bool(cache_cfg.get("overwrite", True))
            total_records = _merge_rank_caches(
                work_root=work_root,
                final_root=final_root,
                manifest=manifest,
                world_size=info.world_size,
                overwrite=overwrite,
            )
            print(
                f"saved {total_records:,} phase/effect records to {final_root}"
            )

            # merged_root has been moved out; remove leftover rank metadata.
            if work_root.exists():
                shutil.rmtree(work_root)

        _barrier(info)

    finally:
        _destroy_process_group(info)


if __name__ == "__main__":
    main()
