"""Stage 1 CTE training on RoboTwin with canonical semantic task identity."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from fastwam.datasets.zeva_robotwin_dataset import ZevaRobotWinDataset
from fastwam.zeva import (
    CTETrainIndex,
    CausalTransitionEncoder,
    CausalTransitionEncoderConfig,
    CTELossConfig,
    TaskBalancedCTEBatchSampler,
    build_cte_training_index,
    causal_transition_encoder_loss,
)
from fastwam.zeva.checkpoint import load_cte_checkpoint, save_cte_checkpoint
from fastwam.zeva.cte_latent_cache import CachedCTELatentWindowDataset
from fastwam.zeva.schemas import sha256_file
from fastwam.zeva.semantic_tasks import parse_episode_index_from_id
from fastwam.zeva.vae_adapter import FastWAMCTELatentEncoder, load_frozen_wan_vae


def _cfg_dict(value) -> dict:
    return {} if value is None else dict(OmegaConf.to_container(value, resolve=True))


def _identity_collate(batch):
    return batch


def _video_size_hw(cfg: DictConfig) -> tuple[int, int]:
    value = cfg.data.train.get("video_size")
    if value is None or len(value) != 2:
        raise ValueError("data.train.video_size must be [H, W] for Zeva CTE training")
    size = tuple(int(v) for v in value)
    if min(size) < 1:
        raise ValueError(f"data.train.video_size must be positive, got {size}")
    return size


def _distributed_context() -> tuple[int, int, int, bool]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError(f"invalid distributed environment: rank={rank}, world_size={world_size}")
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    return rank, local_rank, world_size, rank == 0


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def _reduce_mean(values: list[float], device: torch.device, world_size: int) -> list[float]:
    if world_size == 1:
        return values
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(world_size)
    return [float(v) for v in tensor.cpu()]


def _semantic_remap_training_index(
    rows: list[CTETrainIndex],
    base_dataset,
) -> list[CTETrainIndex]:
    """Replace LeRobot language task_index with canonical RoboTwin task id.

    ``build_cte_training_index`` intentionally uses a metadata-only fast path
    and therefore never calls RobotVideoDataset._get().  Remapping here keeps
    that fast path while ensuring the sampler and task contrastive loss see
    canonical semantic tasks.
    """
    resolver = getattr(base_dataset, "resolve_semantic_task_id", None)
    if resolver is None:
        raise RuntimeError(
            "RobotVideoDataset must expose resolve_semantic_task_id; apply the semantic-task dataset patch"
        )

    remapped: list[CTETrainIndex] = []
    for row in rows:
        episode_index = parse_episode_index_from_id(row.episode_id)
        semantic_task = resolver(
            episode_index=episode_index,
            raw_task_index=row.task_id,
            strict=True,
        )
        remapped.append(
            CTETrainIndex(
                dataset_index=row.dataset_index,
                episode_id=row.episode_id,
                task_id=str(semantic_task),
                episode_step=row.episode_step,
            )
        )
    return remapped


def _validate_semantic_index(rows: list[CTETrainIndex], cfg: DictConfig) -> dict:
    by_task: dict[str, set[str]] = defaultdict(set)
    windows_per_task: Counter[str] = Counter()
    for row in rows:
        task = str(row.task_id)
        by_task[task].add(str(row.episode_id))
        windows_per_task[task] += 1

    task_count = len(by_task)
    episodes_per_task = {task: len(episodes) for task, episodes in by_task.items()}
    singleton_tasks = sum(count == 1 for count in episodes_per_task.values())
    min_episodes = min(episodes_per_task.values()) if episodes_per_task else 0
    max_episodes = max(episodes_per_task.values()) if episodes_per_task else 0

    expected = cfg.get("cte_expected_semantic_task_count")
    if expected not in (None, "", "None", "null") and task_count != int(expected):
        raise ValueError(
            f"semantic task count mismatch: expected {int(expected)}, got {task_count}. "
            "Do not train until the episode->canonical-task mapping is fixed."
        )

    minimum = int(cfg.get("cte_min_episodes_per_semantic_task") or 2)
    bad = sorted((task, count) for task, count in episodes_per_task.items() if count < minimum)
    if bad:
        raise ValueError(
            f"semantic task map is too sparse: {len(bad)} tasks have < {minimum} episodes; "
            f"examples={bad[:20]}"
        )

    stats = {
        "semantic_task_count": task_count,
        "singleton_semantic_tasks": singleton_tasks,
        "min_episodes_per_task": min_episodes,
        "max_episodes_per_task": max_episodes,
        "episodes_per_task": dict(sorted(episodes_per_task.items())),
        "windows_per_task": dict(sorted(windows_per_task.items())),
    }
    return stats


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    rank, local_rank, world_size, is_main_process = _distributed_context()
    metrics_file = None
    try:
        configured_device = cfg.get("device")
        if configured_device is None or str(configured_device).strip().lower() in {"", "none", "null"}:
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_name = str(configured_device)
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Stage 1 requested CUDA but CUDA is unavailable")
        if device_name == "cuda":
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device(device_name)
            if device.type == "cuda":
                torch.cuda.set_device(device)

        seed = int(cfg.get("seed", 42))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        stats_value = str(cfg.data.train.get("pretrained_norm_stats", ""))
        if stats_value in {"", "None", "null"} or not Path(stats_value).is_file():
            raise FileNotFoundError(
                "Zeva Stage 1 requires an existing data.train.pretrained_norm_stats file"
            )

        zeva = cfg.model.get("zeva", {})
        cte_values = _cfg_dict(zeva.get("cte"))
        latent_cache_value = cte_values.get("latent_cache_path")
        use_latent_cache = latent_cache_value not in (None, "", "None", "null")

        allowed = set(CausalTransitionEncoderConfig.__dataclass_fields__)
        model = CausalTransitionEncoder(
            CausalTransitionEncoderConfig(**{k: v for k, v in cte_values.items() if k in allowed})
        )
        cte_input_type = str(cte_values.get("input_type", "rgb_frame"))
        if cte_input_type not in {"rgb_frame", "wan_vae_latent"}:
            raise ValueError("zeva.cte.input_type must be rgb_frame or wan_vae_latent")
        if cte_input_type == "rgb_frame" and model.cfg.image_channels != 3:
            raise ValueError("rgb_frame CTE training requires image_channels=3")

        cte_vae_input_size = _video_size_hw(cfg)
        frame_encoder = None
        vae_metadata: dict[str, object] = {}
        base = None

        if use_latent_cache:
            if cte_input_type != "wan_vae_latent":
                raise ValueError(
                    "model.zeva.cte.latent_cache_path requires input_type=wan_vae_latent"
                )

            semantic_map_value = str(cfg.data.train.get("semantic_task_map_path", ""))
            if semantic_map_value in {"", "None", "null"} or not Path(semantic_map_value).is_file():
                raise FileNotFoundError(
                    "cached Stage-1 training still requires semantic_task_map_path "
                    "for strict cache identity validation"
                )

            dataset = CachedCTELatentWindowDataset(
                str(latent_cache_value),
                expected_dataset_stats_sha256=sha256_file(stats_value),
                expected_semantic_task_sha256=sha256_file(semantic_map_value),
                expected_video_size=cte_vae_input_size,
                expected_action_dim=model.cfg.action_dim,
                expected_transition_steps=model.cfg.transition_steps,
                expected_latent_channels=model.cfg.image_channels,
            )
            semantic_identity = dataset.semantic_task_identity
            vae_metadata = dict(dataset.vae_metadata)

            if tuple(dataset.cte_vae_input_size) != tuple(cte_vae_input_size):
                raise ValueError(
                    "CTE latent cache / training VAE input-size mismatch: "
                    f"cache={dataset.cte_vae_input_size}, train={cte_vae_input_size}"
                )

            if is_main_process:
                print("========== CTE input ==========")
                print(f"precomputed latent cache: {Path(str(latent_cache_value)).resolve()}")
                print(f"cached windows: {len(dataset):,}")
                print(f"latent shape: {dataset.latent_shape}")
                print("RGB decode: DISABLED")
                print("Wan VAE in training: DISABLED")
        else:
            base = instantiate(cfg.data.train)
            if not bool(getattr(base, "require_semantic_task_id", False)):
                raise ValueError(
                    "Formal semantic CTE training requires data.train.require_semantic_task_id=true"
                )
            semantic_identity = getattr(base, "semantic_task_identity", None)
            if semantic_identity is None:
                raise ValueError("formal semantic CTE training requires semantic_task_map_path")

            dataset = ZevaRobotWinDataset(base)
            if len(dataset) == 0:
                raise ValueError("Stage 1 dataset is empty")

            if cte_input_type == "wan_vae_latent":
                model_values = _cfg_dict(cfg.model)
                vae, vae_metadata = load_frozen_wan_vae(
                    model_id=str(model_values.get("model_id", "Wan-AI/Wan2.2-TI2V-5B")),
                    tokenizer_model_id=str(
                        model_values.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B")
                    ),
                    device=str(device),
                    torch_dtype=torch.float32 if device.type == "cpu" else torch.bfloat16,
                    redirect_common_files=bool(model_values.get("redirect_common_files", True)),
                )
                frame_encoder = FastWAMCTELatentEncoder(
                    vae,
                    resize=cte_vae_input_size,
                    expected_channels=model.cfg.image_channels,
                    input_range="minus_one_one",
                ).encode_history

        if (
            model.cfg.action_dim != 14
            or model.cfg.transition_steps != 4
            or model.cfg.effect_window_transitions != 4
        ):
            raise ValueError(
                "RoboTwin Zeva Stage 1 requires action_dim=14, transition_steps=4, "
                "effect_window_transitions=4"
            )

        model.to(device)
        optimizer = AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=float(cfg.get("learning_rate", 2e-4)),
            weight_decay=float(cfg.get("weight_decay", 0.0)),
        )

        batch_size = max(int(cfg.get("batch_size") or 1), 1)
        sample_stride = int(cfg.data.train.get("global_sample_stride", 1))
        if sample_stride != 1:
            raise ValueError("RoboTwin Zeva requires data.train.global_sample_stride=1")
        steps = int(cfg.get("max_steps") or 1000)

        if use_latent_cache:
            training_index = list(dataset.training_index)
            if not training_index:
                raise ValueError("CTE latent cache contains no training windows")
        else:
            raw_training_index = build_cte_training_index(
                dataset,
                source_window_actions=32,
                sample_stride=sample_stride,
            )
            if not raw_training_index:
                raise ValueError("Stage 1 contains no complete, non-overlapping CTE windows")
            assert base is not None
            training_index = _semantic_remap_training_index(raw_training_index, base)

        semantic_stats = _validate_semantic_index(training_index, cfg)
        semantic_tasks = tuple(sorted({str(row.task_id) for row in training_index}))
        semantic_lookup = {task: index for index, task in enumerate(semantic_tasks)}

        if is_main_process:
            print("========== Semantic CTE identity ==========")
            print(f"semantic map: {semantic_identity['path']}")
            print(f"semantic map sha256: {semantic_identity['sha256']}")
            print(f"semantic tasks: {semantic_stats['semantic_task_count']}")
            print(f"min episodes/task: {semantic_stats['min_episodes_per_task']}")
            print(f"max episodes/task: {semantic_stats['max_episodes_per_task']}")
            print("semantic task identity: PASSED")

        samples_per_task = int(cfg.get("cte_samples_per_task") or 4)
        configured_tasks = cfg.get("cte_tasks_per_batch")
        tasks_per_batch = (
            None
            if configured_tasks in (None, "", "None", "null")
            else int(configured_tasks)
        )
        if tasks_per_batch is None:
            tasks_per_batch = max(1, min(4, batch_size // max(samples_per_task, 1)))

        batch_sampler = TaskBalancedCTEBatchSampler(
            training_index,
            batch_size=batch_size,
            tasks_per_batch=tasks_per_batch,
            samples_per_task=samples_per_task,
            seed=seed,
        )

        scheduler = CosineAnnealingLR(optimizer, T_max=max(steps, 1))
        model.train()
        step = 0
        train_model = model
        if world_size > 1:
            train_model = DistributedDataParallel(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
                find_unused_parameters=True,
            )

        output_dir = Path(str(cfg.output_dir))
        if is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
        _barrier(world_size)

        if is_main_process:
            OmegaConf.save(config=cfg, f=str(output_dir / "config.yaml"), resolve=True)
            dataset_manifest = {
                "dataset_dirs": [str(v) for v in cfg.data.train.dataset_dirs],
                "dataset_stats": stats_value,
                "dataset_stats_sha256": sha256_file(stats_value),
                "semantic_task_identity": semantic_identity,
                "semantic_task_stats": semantic_stats,
                "action_dim": model.cfg.action_dim,
                "transition_steps": model.cfg.transition_steps,
                "action_horizon": model.cfg.transition_steps * 8,
                "video_frames": 9,
                "image_channels": model.cfg.image_channels,
                "cte_input_type": cte_input_type,
                "latent_channels": model.cfg.image_channels if cte_input_type == "wan_vae_latent" else 0,
                "vae_metadata": vae_metadata,
                "cte_vae_input_size": list(cte_vae_input_size),
                "camera_keys": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
                "cte_latent_cache": (
                    str(Path(str(latent_cache_value)).resolve())
                    if use_latent_cache
                    else None
                ),
                "cte_latent_cache_manifest_sha256": (
                    sha256_file(Path(str(latent_cache_value)) / "manifest.json")
                    if use_latent_cache
                    else None
                ),
            }
            (output_dir / "dataset_manifest.json").write_text(
                json.dumps(dataset_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (output_dir / "semantic_task_stats.json").write_text(
                json.dumps(semantic_stats, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        _barrier(world_size)

        resume_path = cfg.get("resume")
        if resume_path not in (None, "", "None", "null"):
            resume_payload = load_cte_checkpoint(
                str(resume_path),
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location=str(device),
            )
            old_identity = resume_payload.get("semantic_task_identity") or {}
            if old_identity.get("sha256") != semantic_identity.get("sha256"):
                raise ValueError(
                    "resume checkpoint semantic task map hash differs from current mapping"
                )
            step = int(resume_payload.get("step", 0))
            metrics_file = (
                (output_dir / "metrics.jsonl").open("a", encoding="utf-8")
                if is_main_process
                else None
            )
        else:
            metrics_file = (
                (output_dir / "metrics.jsonl").open("w", encoding="utf-8")
                if is_main_process
                else None
            )

        def train_batch(batch: list[dict]) -> dict:
            frames_list, actions_list, valid_list, transition_valid_list = [], [], [], []
            preencoded_flags: list[bool] = []
            semantic_ids: list[int] = []
            batch_task_keys: list[str] = []
            batch_episode_ids: list[str] = []
            batch_window_starts: list[int] = []

            for sample in batch:
                frames = sample.get("cte_frames")
                preencoded = frames is not None
                if frames is None:
                    frames = torch.cat(
                        (sample["before_frames"], sample["after_frames"][-1:]), dim=0
                    )
                elif frames.ndim != 4 or frames.shape[0] != 9:
                    raise ValueError("sample['cte_frames'] must be [9,C,H,W]")

                frames_list.append(frames)
                preencoded_flags.append(preencoded)
                actions_list.append(sample["transition_actions"])
                valid_list.append(sample["frame_valid"])
                transition_valid_list.append(sample["transition_valid"])

                episode = sample["episode"]
                task_key = str(episode.task_id)
                if task_key not in semantic_lookup:
                    raise KeyError(f"unexpected semantic task in batch: {task_key}")
                semantic_ids.append(semantic_lookup[task_key])
                batch_task_keys.append(task_key)
                batch_episode_ids.append(str(episode.episode_id))
                batch_window_starts.append(int(episode.episode_step))

            frames = torch.stack(frames_list, dim=0)
            if any(preencoded_flags) and not all(preencoded_flags):
                raise ValueError("CTE batch cannot mix RGB and pre-encoded latent frames")
            if frame_encoder is not None and not all(preencoded_flags):
                frames = frame_encoder(frames)

            if all(preencoded_flags):
                # Cached values are stored as exact BF16 bit patterns. The
                # online VAE path historically returns float32 to CTE, so cast
                # after H2D to preserve the original Stage-1 numeric contract.
                frames = frames.to(
                    device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
            else:
                frames = frames.to(device, non_blocking=True)

            actions = torch.stack(actions_list, dim=0).to(
                device,
                non_blocking=True,
            )
            valid = torch.stack(valid_list, dim=0).to(
                device,
                non_blocking=True,
            )
            transition_valid = torch.stack(
                transition_valid_list,
                dim=0,
            ).to(
                device,
                non_blocking=True,
            )

            output = train_model(
                frames,
                actions,
                valid_mask=valid,
                transition_valid=transition_valid,
            )
            losses = causal_transition_encoder_loss(
                output,
                actions,
                valid,
                torch.tensor(semantic_ids, dtype=torch.long, device=device),
                CTELossConfig(),
            )
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(
                    f"non-finite CTE loss at step {step + 1}: {losses['total'].item()}"
                )

            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                train_model.parameters(), float(cfg.get("max_grad_norm", 1.0))
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"non-finite CTE gradient norm at step {step + 1}: {grad_norm.item()}"
                )
            optimizer.step()
            scheduler.step()
            model.update_ema_target()

            reduced = _reduce_mean(
                [
                    float(losses["total"].detach()),
                    float(losses["action"].detach()),
                    float(losses["vision"].detach()),
                    float(losses["loss_task"].detach()),
                    float(losses["loss_phase"].detach()),
                    float(losses["loss_effect"].detach()),
                    float(grad_norm.detach()),
                ],
                device,
                world_size,
            )

            counts = Counter(batch_task_keys)
            metric_payload = {
                "step": int(step + 1),
                "loss": reduced[0],
                "action": reduced[1],
                "vision": reduced[2],
                "task": reduced[3],
                "phase": reduced[4],
                "effect": reduced[5],
                "grad_norm": reduced[6],
                "actual_batch_size": len(batch) * world_size,
                "distinct_task_count": len(counts),
                "task_positive_anchor_count": sum(c for c in counts.values() if c > 1),
                "task_positive_pair_count": sum(c * (c - 1) // 2 for c in counts.values()),
                "distinct_episode_count": len(set(batch_episode_ids)),
                "dataset_index": [int(sample.get("dataset_index", -1)) for sample in batch],
                "episode_id": batch_episode_ids,
                "task_id": batch_task_keys,
                "episode_step": batch_window_starts,
            }
            metric_payload.update(
                {
                    "cte/loss": metric_payload["loss"],
                    "cte/loss_action": metric_payload["action"],
                    "cte/loss_vision": metric_payload["vision"],
                    "cte/loss_task": metric_payload["task"],
                    "cte/loss_phase": metric_payload["phase"],
                    "cte/loss_effect": metric_payload["effect"],
                    "cte/actual_batch_size": metric_payload["actual_batch_size"],
                    "cte/distinct_task_count": metric_payload["distinct_task_count"],
                    "cte/task_positive_anchor_count": metric_payload["task_positive_anchor_count"],
                    "cte/task_positive_pair_count": metric_payload["task_positive_pair_count"],
                    "cte/distinct_episode_count": metric_payload["distinct_episode_count"],
                }
            )
            if is_main_process:
                assert metrics_file is not None
                metrics_file.write(json.dumps(metric_payload) + "\n")
                metrics_file.flush()
            return metric_payload

        epoch = 0
        while step < steps:
            batch_sampler.set_epoch(epoch)
            epoch += 1
            all_batches = list(batch_sampler)
            if not all_batches:
                raise RuntimeError("Stage 1 sampler produced no training batch")

            if world_size > 1:
                batches_per_rank = (len(all_batches) + world_size - 1) // world_size
                padded_batches = all_batches + [all_batches[0]] * (
                    batches_per_rank * world_size - len(all_batches)
                )
                local_batches = padded_batches[rank::world_size]
            else:
                local_batches = all_batches

            local_index_batches = [
                [int(row.dataset_index) for row in index_batch]
                for index_batch in local_batches
            ]
            num_workers = max(int(cfg.get("num_workers") or 0), 0)
            loader_kwargs = {
                "dataset": dataset,
                "batch_sampler": local_index_batches,
                "num_workers": num_workers,
                "collate_fn": _identity_collate,
                "pin_memory": device.type == "cuda",
            }
            if num_workers > 0:
                loader_kwargs["prefetch_factor"] = 2
                # Particularly useful for mmap latent-cache workers: keep the
                # per-worker mmap handles alive instead of reopening them.
                loader_kwargs["persistent_workers"] = True
            train_loader = DataLoader(**loader_kwargs)

            remaining_steps = steps - step
            iterator = tqdm(
                train_loader,
                total=min(len(local_index_batches), remaining_steps),
                desc=f"CTE training rank {rank}",
                disable=not is_main_process,
                dynamic_ncols=True,
            )
            progressed = False
            for batch in iterator:
                if step >= steps:
                    break
                metric = train_batch(batch)
                step += 1
                progressed = True
                if is_main_process:
                    iterator.set_postfix(
                        step=step,
                        loss=f"{metric['loss']:.4f}",
                        task=f"{metric['task']:.3f}",
                    )
            if not progressed:
                raise RuntimeError("Stage 1 sampler produced no training batch")

        if metrics_file is not None:
            metrics_file.close()
            metrics_file = None
        _barrier(world_size)

        if is_main_process:
            checkpoint_config = _cfg_dict(zeva)
            checkpoint_config["semantic_task_identity"] = semantic_identity
            save_cte_checkpoint(
                output_dir / "cte.pt",
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                config=checkpoint_config,
                cte_input_type=cte_input_type,
                vae_metadata=vae_metadata,
                cte_vae_input_size=cte_vae_input_size,
            )
            # Add explicit top-level identity for cheap compatibility checks.
            payload = torch.load(output_dir / "cte.pt", map_location="cpu", weights_only=False)
            payload["semantic_task_identity"] = semantic_identity
            payload["semantic_task_stats"] = semantic_stats
            torch.save(payload, output_dir / "cte.pt")
            print(f"saved Stage 1 checkpoint: {output_dir / 'cte.pt'}")

        _barrier(world_size)

    finally:
        if metrics_file is not None:
            metrics_file.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
