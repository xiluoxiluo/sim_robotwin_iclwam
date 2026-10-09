"""Precompute RoboTwin Stage-1 Wan-VAE latent windows with torchrun.

Recommended launch on one 16-GPU node:

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m torch.distributed.run \
  --standalone --nproc_per_node=16 \
  scripts/precompute_zeva_cte_latents.py \
  task=robotwin_zeva_fastwam_3cam_384 \
  +model.zeva.cte.latent_cache_path=/ABS/PATH/zeva_cte_latent_cache_v1 \
  +cte_latent_cache.batch_size=4 \
  +cte_latent_cache.num_workers=2 \
  +cte_latent_cache.flush_every_batches=8

Design:
- force PyAV; TorchCodec is never attempted;
- use the exact RobotVideoDataset preprocessing used by Stage-1;
- use only the non-overlapping 32-action Stage-1 windows;
- assign whole episodes to ranks: episode_index % world_size;
- store latent values losslessly as raw BF16 bits in mmap .npy files;
- also cache normalized actions/masks, so Stage-1 training needs neither RGB
  decoding nor the frozen Wan VAE;
- resumable per rank via progress.json.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from fastwam.datasets.zeva_robotwin_dataset import ZevaRobotWinDataset
from fastwam.zeva.cte_latent_cache import (
    CACHE_FORMAT,
    LATENT_STORAGE_DTYPE,
    bf16_tensor_to_u16_numpy,
)
from fastwam.zeva.schemas import sha256_file
from fastwam.zeva.semantic_tasks import parse_episode_index_from_id
from fastwam.zeva.stage1_sampling import CTETrainIndex, build_cte_training_index
from fastwam.zeva.vae_adapter import FastWAMCTELatentEncoder, load_frozen_wan_vae


def _cfg_dict(value) -> dict:
    return {} if value is None else dict(OmegaConf.to_container(value, resolve=True))


def _video_size_hw(cfg: DictConfig) -> tuple[int, int]:
    value = cfg.data.train.get("video_size")
    if value is None or len(value) != 2:
        raise ValueError("data.train.video_size must be [H, W]")
    size = tuple(int(v) for v in value)
    if min(size) < 1:
        raise ValueError(f"invalid video_size: {size}")
    return size


def _distributed_context() -> tuple[int, int, int, bool]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size < 1:
        raise ValueError(
            f"invalid WORLD_SIZE={world_size}"
        )

    if rank < 0 or rank >= world_size:
        raise ValueError(
            f"invalid rank={rank}, world_size={world_size}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "VAE latent precompute requires CUDA"
        )

    visible_gpu_count = torch.cuda.device_count()

    if local_rank < 0 or local_rank >= visible_gpu_count:
        raise RuntimeError(
            "LOCAL_RANK exceeds visible CUDA devices: "
            f"local_rank={local_rank}, "
            f"visible_gpu_count={visible_gpu_count}, "
            f"CUDA_VISIBLE_DEVICES="
            f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
        )

    # Important: bind rank -> GPU BEFORE NCCL initialization.
    torch.cuda.set_device(local_rank)

    device = torch.device(
        "cuda",
        local_rank,
    )

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=device,
        )

    return (
        rank,
        local_rank,
        world_size,
        rank == 0,
    )


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _identity_collate(batch):
    return batch


def _semantic_remap_rows(rows: list[CTETrainIndex], base_dataset) -> list[CTETrainIndex]:
    resolver = getattr(base_dataset, "resolve_semantic_task_id", None)
    if not callable(resolver):
        raise RuntimeError("RobotVideoDataset.resolve_semantic_task_id() is required")
    out: list[CTETrainIndex] = []
    for row in rows:
        episode_index = parse_episode_index_from_id(row.episode_id)
        semantic_task = resolver(
            episode_index=episode_index,
            raw_task_index=row.task_id,
            strict=True,
        )
        out.append(
            CTETrainIndex(
                dataset_index=int(row.dataset_index),
                episode_id=str(row.episode_id),
                task_id=str(semantic_task),
                episode_step=int(row.episode_step),
            )
        )
    return out


def _semantic_stats(rows: list[CTETrainIndex]) -> dict[str, Any]:
    by_task: dict[str, set[str]] = defaultdict(set)
    windows_per_task: Counter[str] = Counter()
    for row in rows:
        task = str(row.task_id)
        by_task[task].add(str(row.episode_id))
        windows_per_task[task] += 1
    episodes_per_task = {task: len(value) for task, value in by_task.items()}
    return {
        "semantic_task_count": len(by_task),
        "singleton_semantic_tasks": sum(v == 1 for v in episodes_per_task.values()),
        "min_episodes_per_task": min(episodes_per_task.values()) if episodes_per_task else 0,
        "max_episodes_per_task": max(episodes_per_task.values()) if episodes_per_task else 0,
        "episodes_per_task": dict(sorted(episodes_per_task.items())),
        "windows_per_task": dict(sorted(windows_per_task.items())),
    }


@dataclass(frozen=True)
class _BuildItem:
    dataset_index: int
    episode_id: str
    episode_step: int
    frames: torch.Tensor
    transition_actions: torch.Tensor
    frame_valid: torch.Tensor
    transition_valid: torch.Tensor
    task_id: str


class _WindowSourceDataset(Dataset):
    """Materialize exact Stage-1 windows from the normal RGB dataset."""

    def __init__(self, dataset: ZevaRobotWinDataset, rows: list[CTETrainIndex]) -> None:
        self.dataset = dataset
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> _BuildItem:
        row = self.rows[int(index)]
        sample = self.dataset[int(row.dataset_index)]
        source_index = int(sample.get("dataset_index", -1))
        if source_index != int(row.dataset_index):
            raise RuntimeError(
                f"cache source mismatch: requested {row.dataset_index}, got {source_index}"
            )

        episode = sample["episode"]
        if str(episode.episode_id) != str(row.episode_id):
            raise ValueError(
                f"episode mismatch for dataset_index={row.dataset_index}: "
                f"index={row.episode_id}, sample={episode.episode_id}"
            )
        if int(episode.episode_step) != int(row.episode_step):
            raise ValueError(
                f"episode_step mismatch for dataset_index={row.dataset_index}: "
                f"index={row.episode_step}, sample={episode.episode_step}"
            )

        frames = sample.get("cte_frames")
        if frames is None:
            frames = torch.cat(
                (sample["before_frames"], sample["after_frames"][-1:]),
                dim=0,
            )
        if frames.ndim != 4 or frames.shape[0] != 9 or frames.shape[1] != 3:
            raise ValueError(
                "precompute source must be RGB [9,3,H,W]; "
                f"got {tuple(frames.shape)}"
            )

        return _BuildItem(
            dataset_index=int(row.dataset_index),
            episode_id=str(episode.episode_id),
            episode_step=int(episode.episode_step),
            frames=frames,
            transition_actions=sample["transition_actions"].float(),
            frame_valid=sample["frame_valid"].bool(),
            transition_valid=sample["transition_valid"].bool(),
            task_id=str(episode.task_id),
        )


def _load_progress(rank_dir: Path, local_count: int) -> int:
    path = rank_dir / "progress.json"
    if not path.is_file():
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("total", -1)) != int(local_count):
        raise ValueError(
            f"resume count mismatch in {path}: expected {local_count}, "
            f"got {payload.get('total')}"
        )
    next_offset = int(payload.get("next_offset", 0))
    if next_offset < 0 or next_offset > local_count:
        raise ValueError(f"invalid resume offset {next_offset} in {path}")
    return next_offset


def _open_existing_arrays(rank_dir: Path) -> dict[str, np.ndarray]:
    return {
        "latents": np.load(rank_dir / "latents_bf16_u16.npy", mmap_mode="r+"),
        "actions": np.load(rank_dir / "transition_actions.npy", mmap_mode="r+"),
        "frame_valid": np.load(rank_dir / "frame_valid.npy", mmap_mode="r+"),
        "transition_valid": np.load(rank_dir / "transition_valid.npy", mmap_mode="r+"),
    }


def _create_arrays(
    rank_dir: Path,
    *,
    count: int,
    latent_shape: tuple[int, ...],
    action_shape: tuple[int, ...],
    frame_valid_shape: tuple[int, ...],
    transition_valid_shape: tuple[int, ...],
) -> dict[str, np.ndarray]:
    return {
        "latents": np.lib.format.open_memmap(
            rank_dir / "latents_bf16_u16.npy",
            mode="w+",
            dtype=np.uint16,
            shape=(count, *latent_shape),
        ),
        "actions": np.lib.format.open_memmap(
            rank_dir / "transition_actions.npy",
            mode="w+",
            dtype=np.float32,
            shape=(count, *action_shape),
        ),
        "frame_valid": np.lib.format.open_memmap(
            rank_dir / "frame_valid.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(count, *frame_valid_shape),
        ),
        "transition_valid": np.lib.format.open_memmap(
            rank_dir / "transition_valid.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(count, *transition_valid_shape),
        ),
    }


def _flush_arrays(arrays: dict[str, np.ndarray]) -> None:
    for value in arrays.values():
        value.flush()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    rank, local_rank, world_size, is_main = _distributed_context()
    device = torch.device("cuda", local_rank)

    try:
        zeva = cfg.model.get("zeva", {})
        cte_values = _cfg_dict(zeva.get("cte"))
        cache_cfg = _cfg_dict(cfg.get("cte_latent_cache"))

        cache_value = cache_cfg.get("path", cte_values.get("latent_cache_path"))
        if cache_value in (None, "", "None", "null"):
            raise ValueError(
                "Set +model.zeva.cte.latent_cache_path=/absolute/cache/path "
                "or +cte_latent_cache.path=/absolute/cache/path"
            )
        cache_root = Path(str(cache_value)).expanduser().resolve()

        batch_size = max(int(cache_cfg.get("batch_size", 4)), 1)
        num_workers = max(int(cache_cfg.get("num_workers", 2)), 0)
        prefetch_factor = max(int(cache_cfg.get("prefetch_factor", 2)), 1)
        flush_every = max(int(cache_cfg.get("flush_every_batches", 8)), 1)
        overwrite = bool(cache_cfg.get("overwrite", False))
        presample_images = bool(cache_cfg.get("presample_images", True))

        stats_path = Path(str(cfg.data.train.get("pretrained_norm_stats", "")))
        if not stats_path.is_file():
            raise FileNotFoundError(
                "VAE latent cache requires data.train.pretrained_norm_stats"
            )
        semantic_map_path = Path(str(cfg.data.train.get("semantic_task_map_path", "")))
        if not semantic_map_path.is_file():
            raise FileNotFoundError(
                "VAE latent cache requires data.train.semantic_task_map_path"
            )

        # Force the exact requested decoder.  This bypasses get_safe_default_codec
        # and therefore never attempts TorchCodec.
        OmegaConf.update(cfg, "data.train.video_backend", "pyav", force_add=True)
        OmegaConf.update(cfg, "data.train.use_text_embed_cache", False, force_add=True)

        # Only request the 9 image timestamps needed by the 32-action CTE
        # window. PyAV still decodes inter-frame dependencies internally, but
        # this avoids carrying/resize-processing 33 RGB tensors in Python.
        BaseLerobotDataset.presample_images = presample_images

        if rank == 0 and overwrite and cache_root.exists():
            shutil.rmtree(cache_root)
        _barrier(world_size)
        cache_root.mkdir(parents=True, exist_ok=True)

        base = instantiate(cfg.data.train)
        if not bool(getattr(base, "require_semantic_task_id", False)):
            raise ValueError("formal cache generation requires require_semantic_task_id=true")
        semantic_identity = getattr(base, "semantic_task_identity", None)
        if semantic_identity is None:
            raise ValueError("semantic_task_identity is missing")
        dataset = ZevaRobotWinDataset(base)

        sample_stride = int(cfg.data.train.get("global_sample_stride", 1))
        if sample_stride != 1:
            raise ValueError("RoboTwin Zeva V1 requires global_sample_stride=1")

        all_rows = build_cte_training_index(
            dataset,
            source_window_actions=32,
            sample_stride=sample_stride,
        )
        if not all_rows:
            raise RuntimeError("no Stage-1 CTE windows found")

        # Keep all windows from one episode on one GPU.  This is deterministic
        # and prevents duplicate window generation across ranks.
        local_rows = [
            row
            for row in all_rows
            if parse_episode_index_from_id(row.episode_id) % world_size == rank
        ]
        local_count = len(local_rows)
        rank_dir = cache_root / f"rank_{rank:03d}"
        rank_dir.mkdir(parents=True, exist_ok=True)

        if (rank_dir / "DONE").is_file() and (rank_dir / "rank_meta.json").is_file():
            rank_meta = json.loads((rank_dir / "rank_meta.json").read_text(encoding="utf-8"))
            if int(rank_meta.get("num_windows", -1)) != local_count:
                raise ValueError(
                    f"rank {rank} DONE cache count mismatch: "
                    f"expected {local_count}, got {rank_meta.get('num_windows')}"
                )
            print(
                f"[latent-cache rank {rank}] already complete: "
                f"{local_count:,} windows"
            )
        else:
            progress_offset = _load_progress(rank_dir, local_count)
            existing_meta_path = rank_dir / "rank_meta.json"
            arrays: dict[str, np.ndarray] | None = None
            latent_shape: tuple[int, ...] | None = None

            if progress_offset > 0:
                if not existing_meta_path.is_file():
                    raise FileNotFoundError(
                        f"resume progress exists but rank_meta.json is missing: {rank_dir}"
                    )
                rank_meta = json.loads(existing_meta_path.read_text(encoding="utf-8"))
                latent_shape = tuple(int(v) for v in rank_meta["latent_shape"])
                arrays = _open_existing_arrays(rank_dir)
                if arrays["latents"].shape[0] != local_count:
                    raise ValueError("resume mmap length does not match current local window count")

            # Do not load the Wan VAE until after resume checks.
            model_values = _cfg_dict(cfg.model)
            vae, vae_metadata = load_frozen_wan_vae(
                model_id=str(model_values.get("model_id", "Wan-AI/Wan2.2-TI2V-5B")),
                tokenizer_model_id=str(
                    model_values.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B")
                ),
                device=str(device),
                torch_dtype=torch.bfloat16,
                redirect_common_files=bool(model_values.get("redirect_common_files", True)),
            )
            expected_channels = int(cte_values.get("image_channels", 48))
            cte_vae_input_size = _video_size_hw(cfg)
            frame_encoder = FastWAMCTELatentEncoder(
                vae,
                resize=cte_vae_input_size,
                expected_channels=expected_channels,
                input_range="minus_one_one",
            ).encode_history

            remaining_rows = local_rows[progress_offset:]
            source_dataset = _WindowSourceDataset(dataset, remaining_rows)
            loader_kwargs: dict[str, Any] = {
                "dataset": source_dataset,
                "batch_size": batch_size,
                "shuffle": False,
                "num_workers": num_workers,
                "collate_fn": _identity_collate,
                "pin_memory": True,
            }
            if num_workers > 0:
                loader_kwargs["prefetch_factor"] = prefetch_factor
                loader_kwargs["persistent_workers"] = True
            loader = DataLoader(**loader_kwargs)

            progress = tqdm(
                loader,
                total=(len(remaining_rows) + batch_size - 1) // batch_size,
                desc=f"VAE latent cache GPU{local_rank} rank {rank}/{world_size}",
                unit="batch",
                dynamic_ncols=True,
                position=rank,
                leave=True,
            )

            write_offset = progress_offset
            batches_since_flush = 0
            with torch.inference_mode():
                for items in progress:
                    frames = torch.stack([item.frames for item in items], dim=0)
                    frames = frames.to(device, non_blocking=True)
                    latent = frame_encoder(frames)
                    if latent.ndim != 5 or latent.shape[1] != 9:
                        raise ValueError(
                            "FastWAMCTELatentEncoder must return [B,9,C,H,W], "
                            f"got {tuple(latent.shape)}"
                        )
                    latent_bits = bf16_tensor_to_u16_numpy(latent)

                    actions = torch.stack(
                        [item.transition_actions for item in items], dim=0
                    ).numpy()
                    frame_valid = torch.stack(
                        [item.frame_valid for item in items], dim=0
                    ).to(torch.uint8).numpy()
                    transition_valid = torch.stack(
                        [item.transition_valid for item in items], dim=0
                    ).to(torch.uint8).numpy()

                    if arrays is None:
                        latent_shape = tuple(int(v) for v in latent_bits.shape[1:])
                        arrays = _create_arrays(
                            rank_dir,
                            count=local_count,
                            latent_shape=latent_shape,
                            action_shape=tuple(int(v) for v in actions.shape[1:]),
                            frame_valid_shape=tuple(int(v) for v in frame_valid.shape[1:]),
                            transition_valid_shape=tuple(
                                int(v) for v in transition_valid.shape[1:]
                            ),
                        )
                        rank_meta = {
                            "rank": rank,
                            "world_size": world_size,
                            "num_windows": local_count,
                            "latent_shape": list(latent_shape),
                            "vae_metadata": vae_metadata,
                            "cte_vae_input_size": list(cte_vae_input_size),
                            "latent_storage_dtype": LATENT_STORAGE_DTYPE,
                        }
                        _atomic_json(existing_meta_path, rank_meta)

                    assert arrays is not None
                    count = len(items)
                    end = write_offset + count
                    if end > local_count:
                        raise RuntimeError("rank cache write exceeded allocated window count")
                    arrays["latents"][write_offset:end] = latent_bits
                    arrays["actions"][write_offset:end] = actions
                    arrays["frame_valid"][write_offset:end] = frame_valid
                    arrays["transition_valid"][write_offset:end] = transition_valid
                    write_offset = end
                    batches_since_flush += 1

                    if batches_since_flush >= flush_every:
                        _flush_arrays(arrays)
                        _atomic_json(
                            rank_dir / "progress.json",
                            {"next_offset": write_offset, "total": local_count},
                        )
                        batches_since_flush = 0

                    progress.set_postfix(
                        windows=f"{write_offset:,}/{local_count:,}",
                        refresh=False,
                    )

            if local_count > 0 and arrays is None:
                raise RuntimeError("rank produced no latent batch despite non-empty row assignment")
            if arrays is not None:
                _flush_arrays(arrays)
            if write_offset != local_count:
                raise RuntimeError(
                    f"rank {rank} cache incomplete: wrote {write_offset}, expected {local_count}"
                )
            _atomic_json(
                rank_dir / "progress.json",
                {"next_offset": local_count, "total": local_count},
            )
            (rank_dir / "DONE").write_text("ok\n", encoding="utf-8")
            print(f"[latent-cache rank {rank}] complete: {local_count:,} windows")

        _barrier(world_size)

        if is_main:
            canonical_rows = _semantic_remap_rows(all_rows, base)
            semantic_stats = _semantic_stats(canonical_rows)

            rank_metas: list[dict[str, Any]] = []
            for rank_id in range(world_size):
                rank_dir = cache_root / f"rank_{rank_id:03d}"
                if not (rank_dir / "DONE").is_file():
                    raise RuntimeError(f"rank {rank_id} did not finish latent cache generation")
                rank_metas.append(
                    json.loads((rank_dir / "rank_meta.json").read_text(encoding="utf-8"))
                )

            nonempty_shapes = {
                tuple(int(v) for v in meta["latent_shape"])
                for meta in rank_metas
                if int(meta.get("num_windows", 0)) > 0
            }
            if len(nonempty_shapes) != 1:
                raise ValueError(f"ranks disagree on latent shape: {sorted(nonempty_shapes)}")
            latent_shape = next(iter(nonempty_shapes))

            rank_counts = [0] * world_size
            index_rows: list[dict[str, Any]] = []
            for row in canonical_rows:
                episode_index = parse_episode_index_from_id(row.episode_id)
                rank_id = episode_index % world_size
                offset = rank_counts[rank_id]
                rank_counts[rank_id] += 1
                index_rows.append(
                    {
                        "dataset_index": int(row.dataset_index),
                        "episode_id": str(row.episode_id),
                        "episode_index": int(episode_index),
                        "episode_step": int(row.episode_step),
                        "task_id": str(row.task_id),
                        "rank": int(rank_id),
                        "offset": int(offset),
                    }
                )

            declared_counts = [int(meta["num_windows"]) for meta in rank_metas]
            if rank_counts != declared_counts:
                raise ValueError(
                    f"rank cache counts do not match deterministic partition: "
                    f"index={rank_counts}, files={declared_counts}"
                )

            first_meta = next(
                meta for meta in rank_metas if int(meta.get("num_windows", 0)) > 0
            )
            vae_metadata = dict(first_meta["vae_metadata"])
            for meta in rank_metas:
                if int(meta.get("num_windows", 0)) == 0:
                    continue
                if dict(meta["vae_metadata"]) != vae_metadata:
                    raise ValueError("ranks disagree on VAE identity metadata")

            bytes_per_window = (
                int(np.prod(latent_shape)) * 2
                + (8 * 4 * 14) * 4
                + 9
                + (8 * 4)
            )
            payload_bytes_estimate = int(len(index_rows) * bytes_per_window)

            manifest = {
                "format": CACHE_FORMAT,
                "dataset_dirs": [str(v) for v in cfg.data.train.dataset_dirs],
                "dataset_stats": str(stats_path.resolve()),
                "dataset_stats_sha256": sha256_file(stats_path),
                "semantic_task_identity": semantic_identity,
                "semantic_task_stats": semantic_stats,
                "semantic_task_map_sha256": sha256_file(semantic_map_path),
                "video_backend": "pyav",
                "presample_images": presample_images,
                "video_size": list(_video_size_hw(cfg)),
                "camera_keys": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
                "source_window_actions": 32,
                "sample_stride": sample_stride,
                "video_frames": 9,
                "transition_count": 8,
                "transition_steps": 4,
                "action_dim": 14,
                "latent_channels": int(latent_shape[1]),
                "latent_shape": list(latent_shape),
                "latent_storage_dtype": LATENT_STORAGE_DTYPE,
                "vae_metadata": vae_metadata,
                "cte_vae_input_size": list(_video_size_hw(cfg)),
                "num_windows": len(index_rows),
                "bytes_per_window_estimate": bytes_per_window,
                "payload_bytes_estimate": payload_bytes_estimate,
                "generation_world_size": world_size,
                "rank_window_counts": rank_counts,
                "split": {
                    "is_training_set": bool(cfg.data.train.get("is_training_set", True)),
                    "val_set_proportion": float(cfg.data.train.get("val_set_proportion", 0.0)),
                    "seed": int(cfg.get("seed", 42)),
                },
            }
            _atomic_json(cache_root / "index.json", index_rows)
            _atomic_json(cache_root / "manifest.json", manifest)

            print("========== VAE latent cache ==========")
            print(f"path: {cache_root}")
            print(f"windows: {len(index_rows):,}")
            print(f"latent shape/window: {latent_shape}")
            print(f"storage: {LATENT_STORAGE_DTYPE}")
            print(
                "payload estimate: "
                f"{payload_bytes_estimate / (1024 ** 3):.2f} GiB"
            )
            print(f"semantic tasks: {semantic_stats['semantic_task_count']}")
            print(f"rank counts: {rank_counts}")
            print("VAE LATENT CACHE: PASSED")

        _barrier(world_size)

    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
