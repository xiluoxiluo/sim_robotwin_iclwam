"""Validate a Zeva RoboTwin phase/effect cache-v4.

Validation levels
-----------------
1) Full structural scan (all index rows and all safetensors shards):
   - manifest/checkpoint/stats/semantic identity compatibility;
   - record ordering and temporal invariants;
   - duplicate query/effect detection;
   - shard keys, shapes, dtypes, offsets, valid flags, NaN/Inf;
   - exact phase/effect record counts when dataset metadata is available.
2) Representation diagnostics on sampled cached features:
   - phase same-task NN retrieval and same-vs-cross task cosine;
   - phase progression inside sampled episodes;
   - effect per-dimension spread/effective rank;
   - adjacent-effect vs shuffled-effect cosine (diagnostic only).
3) Optional deep recomputation (``phase_effect_validation.deep_episodes > 0``):
   rebuild selected full-episode-prefix inputs from the Stage-1 latent cache,
   run the frozen CTE again, and compare every phase/effect vector against the
   stored cache record.

This script is intentionally strict for the formal RoboTwin V1 path.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from safetensors.torch import load_file
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset  # noqa: E402
from fastwam.datasets.zeva_robotwin_dataset import ZevaRobotWinDataset  # noqa: E402
from fastwam.zeva import (  # noqa: E402
    CausalTransitionEncoder,
    CausalTransitionEncoderConfig,
    FastWAMCTELatentEncoder,
    load_frozen_wan_vae,
    validate_vae_metadata,
)
from fastwam.zeva.checkpoint import checkpoint_sha256, load_cte_checkpoint  # noqa: E402
from fastwam.zeva.cte_latent_cache import CachedCTELatentWindowDataset  # noqa: E402
from fastwam.zeva.schemas import CacheManifest, sha256_file  # noqa: E402
from scripts.build_zeva_robotwin_cache import (  # noqa: E402
    _build_episode_plans_metadata_only,
    _compute_episode,
    _video_size_hw,
)
from scripts.build_zeva_robotwin_cache_from_latents import (  # noqa: E402
    _TailSourceDataset,
    _canonicalize_plans,
    _expected_effect_count,
    _materialize_episode,
    _tail_required,
    _validate_plan_alignment,
)


def _cfg_dict(value: Any) -> dict[str, Any]:
    return {} if value is None else dict(OmegaConf.to_container(value, resolve=True))


def _reservoir_add(
    reservoir: list[dict[str, Any]],
    item: dict[str, Any],
    seen: int,
    limit: int,
    rng: random.Random,
) -> None:
    if limit <= 0:
        return
    if len(reservoir) < limit:
        reservoir.append(item)
        return
    j = rng.randrange(seen)
    if j < limit:
        reservoir[j] = item


def _effective_rank(x: torch.Tensor) -> float:
    if x.ndim != 2 or x.shape[0] < 2:
        return 0.0
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    # SVD on [N,128] is cheaper/stabler than materializing an N x N matrix.
    s = torch.linalg.svdvals(x)
    eig = s.square()
    total = float(eig.sum())
    if not math.isfinite(total) or total <= 0.0:
        return 0.0
    p = (eig / eig.sum()).clamp_min(1.0e-12)
    return float(torch.exp(-(p * p.log()).sum()))


def _pair_cosine_stats(
    features: torch.Tensor,
    task_ids: list[str],
    episode_ids: list[str],
    *,
    pair_samples: int,
    seed: int,
) -> tuple[float, float]:
    if len(features) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    z = F.normalize(features.float(), dim=-1)
    same: list[float] = []
    cross: list[float] = []
    n = len(z)
    attempts = 0
    max_attempts = max(pair_samples * 40, 10000)
    while (len(same) < pair_samples or len(cross) < pair_samples) and attempts < max_attempts:
        attempts += 1
        i = rng.randrange(n)
        j = rng.randrange(n - 1)
        if j >= i:
            j += 1
        if episode_ids[i] == episode_ids[j]:
            continue
        value = float(torch.dot(z[i], z[j]))
        if task_ids[i] == task_ids[j]:
            if len(same) < pair_samples:
                same.append(value)
        else:
            if len(cross) < pair_samples:
                cross.append(value)
    return (
        float(sum(same) / len(same)) if same else float("nan"),
        float(sum(cross) / len(cross)) if cross else float("nan"),
    )


def _same_task_nn_top1(
    features: torch.Tensor,
    task_ids: list[str],
    episode_ids: list[str],
    *,
    chunk_size: int = 256,
) -> float:
    if len(features) < 2:
        return float("nan")
    z = F.normalize(features.float(), dim=-1)
    task_lookup = {task: i for i, task in enumerate(sorted(set(task_ids)))}
    task = torch.tensor([task_lookup[v] for v in task_ids], dtype=torch.long)
    episode_lookup = {ep: i for i, ep in enumerate(sorted(set(episode_ids)))}
    episode = torch.tensor([episode_lookup[v] for v in episode_ids], dtype=torch.long)
    correct = 0
    usable = 0
    for start in range(0, len(z), chunk_size):
        end = min(start + chunk_size, len(z))
        sim = z[start:end] @ z.T
        # PIM validation must not retrieve the same episode.
        same_episode = episode[start:end, None].eq(episode[None, :])
        sim.masked_fill_(same_episode, -torch.inf)
        finite = torch.isfinite(sim).any(dim=1)
        if bool(finite.any()):
            best = sim.argmax(dim=1)
            correct += int((task[best][finite] == task[start:end][finite]).sum())
            usable += int(finite.sum())
    return float(correct / usable) if usable else float("nan")


def _load_selected_vectors(
    root: Path,
    rows: list[dict[str, Any]],
    tensor_name: str,
) -> torch.Tensor:
    if not rows:
        return torch.empty((0, 0), dtype=torch.float32)
    by_shard: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for output_pos, row in enumerate(rows):
        by_shard[str(row["shard"])].append((output_pos, int(row["offset"])))
    out: list[torch.Tensor | None] = [None] * len(rows)
    for shard_name, requests in by_shard.items():
        tensors = load_file(str(root / shard_name), device="cpu")
        source = tensors[tensor_name]
        for output_pos, offset in requests:
            out[output_pos] = source[offset].float().clone()
        del tensors
    assert all(value is not None for value in out)
    return torch.stack([value for value in out if value is not None])


def _index_cache_rows(root: Path, sample_phase: int, sample_effect: int, seed: int):
    index_path = root / "episode_index.json"
    started = time.perf_counter()
    rows = json.loads(index_path.read_text(encoding="utf-8"))
    print(f"loaded index rows: {len(rows):,} ({time.perf_counter() - started:.1f}s)")

    rng = random.Random(seed)
    phase_sample: list[dict[str, Any]] = []
    effect_sample: list[dict[str, Any]] = []
    phase_seen = 0
    effect_seen = 0

    type_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    episode_ids: set[str] = set()
    phase_windows: set[int] = set()
    effect_keys: set[tuple[str, int, int]] = set()
    rows_by_shard: dict[str, list[int]] = defaultdict(list)
    effect_lookup: dict[tuple[str, int], int] = {}

    previous_order_key: tuple[Any, ...] | None = None
    record_order = {"effect": 0, "phase_query": 1}

    for row_index, row in enumerate(rows):
        record_type = str(row.get("record_type", ""))
        if record_type not in record_order:
            raise ValueError(f"row {row_index}: unsupported record_type={record_type!r}")
        type_counts[record_type] += 1
        task_counts[str(row["task_id"])] += 1
        episode_id = str(row["episode_id"])
        episode_ids.add(episode_id)

        raw_step = int(row["raw_step"])
        start = int(row["start_raw_step"])
        end = int(row["end_raw_step"])
        episode_step = int(row["episode_step"])
        if min(raw_step, start, end, episode_step) < 0:
            raise ValueError(f"row {row_index}: negative temporal coordinate")
        if any(value % 4 for value in (raw_step, start, end, episode_step)):
            raise ValueError(f"row {row_index}: raw-action coordinates must be multiples of 4")

        if record_type == "phase_query":
            if row.get("window_index") is None:
                raise ValueError(f"row {row_index}: phase_query missing window_index")
            if not (episode_step == raw_step == start == end):
                raise ValueError(f"row {row_index}: phase_query temporal fields disagree")
            window = int(row["window_index"])
            if window in phase_windows:
                raise ValueError(f"duplicate phase_query window_index={window}")
            phase_windows.add(window)
            phase_seen += 1
            _reservoir_add(phase_sample, row, phase_seen, sample_phase, rng)
        else:
            if row.get("window_index") is not None:
                raise ValueError(f"row {row_index}: effect row must have window_index=null")
            if episode_step != start or raw_step != end:
                raise ValueError(f"row {row_index}: effect temporal fields disagree")
            if end - start != 16:
                raise ValueError(
                    f"row {row_index}: effect interval must be 16 raw actions, got {end-start}"
                )
            key = (episode_id, start, end)
            if key in effect_keys:
                raise ValueError(f"duplicate effect interval={key}")
            effect_keys.add(key)
            effect_lookup[(episode_id, start)] = row_index
            effect_seen += 1
            _reservoir_add(effect_sample, row, effect_seen, sample_effect, rng)

        shard = str(row["shard"])
        rows_by_shard[shard].append(row_index)
        order_key = (
            episode_id,
            raw_step,
            record_order[record_type],
            int(row.get("effect_index", 0)),
            -1 if row.get("window_index") is None else int(row["window_index"]),
        )
        if previous_order_key is not None and order_key < previous_order_key:
            raise ValueError(
                f"episode_index.json is not in canonical order at row {row_index}: "
                f"previous={previous_order_key}, current={order_key}"
            )
        previous_order_key = order_key

    return {
        "rows": rows,
        "type_counts": type_counts,
        "task_counts": task_counts,
        "episode_ids": episode_ids,
        "phase_sample": phase_sample,
        "effect_sample": effect_sample,
        "rows_by_shard": rows_by_shard,
        "effect_lookup": effect_lookup,
    }


def _scan_all_shards(
    root: Path,
    manifest: CacheManifest,
    rows: list[dict[str, Any]],
    rows_by_shard: dict[str, list[int]],
) -> dict[str, float]:
    phase_sum = torch.zeros(manifest.phase_dim, dtype=torch.float64)
    phase_sq = torch.zeros(manifest.phase_dim, dtype=torch.float64)
    effect_sum = torch.zeros(manifest.effect_dim, dtype=torch.float64)
    effect_sq = torch.zeros(manifest.effect_dim, dtype=torch.float64)
    phase_count = 0
    effect_count = 0

    shard_names = sorted(rows_by_shard)
    for shard_pos, shard_name in enumerate(shard_names, start=1):
        tensors = load_file(str(root / shard_name), device="cpu")
        if set(tensors) != {"phase_pre", "phase_post", "effect", "valid"}:
            raise ValueError(f"{shard_name}: unexpected tensor keys {sorted(tensors)}")
        n = int(tensors["phase_pre"].shape[0])
        if tuple(tensors["phase_pre"].shape) != (n, manifest.phase_dim):
            raise ValueError(f"{shard_name}: phase_pre shape mismatch")
        if tuple(tensors["phase_post"].shape) != (n, manifest.phase_dim):
            raise ValueError(f"{shard_name}: phase_post shape mismatch")
        if tuple(tensors["effect"].shape) != (n, manifest.effect_dim):
            raise ValueError(f"{shard_name}: effect shape mismatch")
        if tuple(tensors["valid"].shape) != (n,):
            raise ValueError(f"{shard_name}: valid shape mismatch")
        for name in ("phase_pre", "phase_post", "effect"):
            if tensors[name].dtype != torch.float32:
                raise ValueError(f"{shard_name}: {name} dtype={tensors[name].dtype}, expected float32")
            if not bool(torch.isfinite(tensors[name]).all()):
                raise ValueError(f"{shard_name}: {name} contains NaN/Inf")
        if tensors["valid"].dtype != torch.bool or not bool(tensors["valid"].all()):
            raise ValueError(f"{shard_name}: valid must be all-true bool")

        row_indices = rows_by_shard[shard_name]
        if len(row_indices) != n:
            raise ValueError(
                f"{shard_name}: index rows={len(row_indices)} != tensor rows={n}"
            )
        offsets = [int(rows[i]["offset"]) for i in row_indices]
        if sorted(offsets) != list(range(n)):
            raise ValueError(f"{shard_name}: offsets must be exactly 0..{n-1}")

        phase_offsets = [int(rows[i]["offset"]) for i in row_indices if rows[i]["record_type"] == "phase_query"]
        effect_offsets = [int(rows[i]["offset"]) for i in row_indices if rows[i]["record_type"] == "effect"]
        if phase_offsets:
            x = tensors["phase_pre"][phase_offsets].double()
            phase_sum += x.sum(dim=0)
            phase_sq += x.square().sum(dim=0)
            phase_count += len(phase_offsets)
        if effect_offsets:
            x = tensors["effect"][effect_offsets].double()
            effect_sum += x.sum(dim=0)
            effect_sq += x.square().sum(dim=0)
            effect_count += len(effect_offsets)
        del tensors
        if shard_pos % 50 == 0 or shard_pos == len(shard_names):
            print(f"shard scan: {shard_pos}/{len(shard_names)}")

    def mean_per_dim_std(total, total_sq, count: int) -> float:
        if count < 1:
            return float("nan")
        mean = total / count
        var = (total_sq / count - mean.square()).clamp_min(0.0)
        return float(var.sqrt().mean())

    return {
        "phase_count": phase_count,
        "effect_count": effect_count,
        "phase_mean_per_dim_std": mean_per_dim_std(phase_sum, phase_sq, phase_count),
        "effect_mean_per_dim_std": mean_per_dim_std(effect_sum, effect_sq, effect_count),
    }


def _phase_progression_diagnostic(
    root: Path,
    rows: list[dict[str, Any]],
    episode_ids: Iterable[str],
) -> dict[str, float]:
    selected = set(episode_ids)
    phase_rows = [
        row for row in rows
        if row["record_type"] == "phase_query" and str(row["episode_id"]) in selected
    ]
    if not phase_rows:
        return {"increment_abs_mean": float("nan"), "positive_fraction": float("nan"), "margin_fraction": float("nan")}
    features = _load_selected_vectors(root, phase_rows, "phase_pre")
    by_episode: dict[str, list[tuple[int, torch.Tensor]]] = defaultdict(list)
    for row, feature in zip(phase_rows, features, strict=True):
        by_episode[str(row["episode_id"])].append((int(row["raw_step"]), feature))
    increments: list[torch.Tensor] = []
    for values in by_episode.values():
        values.sort(key=lambda item: item[0])
        z = torch.stack([value for _, value in values])
        if len(z) < 3:
            continue
        direction = F.normalize(z[-1] - z[0], dim=0)
        increments.append((z[1:] - z[:-1]) @ direction)
    if not increments:
        return {"increment_abs_mean": float("nan"), "positive_fraction": float("nan"), "margin_fraction": float("nan")}
    inc = torch.cat(increments)
    return {
        "increment_abs_mean": float(inc.abs().mean()),
        "positive_fraction": float((inc > 0).float().mean()),
        "margin_fraction": float((inc >= 0.01).float().mean()),
    }


def _adjacent_effect_diagnostic(
    root: Path,
    rows: list[dict[str, Any]],
    effect_lookup: dict[tuple[str, int], int],
    *,
    pairs: int,
    seed: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    effect_indices = [i for i, row in enumerate(rows) if row["record_type"] == "effect"]
    rng.shuffle(effect_indices)
    for row_index in effect_indices:
        row = rows[row_index]
        nxt_index = effect_lookup.get((str(row["episode_id"]), int(row["end_raw_step"])))
        if nxt_index is not None:
            candidates.append((row, rows[nxt_index]))
            if len(candidates) >= pairs:
                break
    if not candidates:
        return {"adjacent_cosine": float("nan"), "shuffled_cosine": float("nan")}
    first_rows = [a for a, _ in candidates]
    second_rows = [b for _, b in candidates]
    a = F.normalize(_load_selected_vectors(root, first_rows, "effect"), dim=-1)
    b = F.normalize(_load_selected_vectors(root, second_rows, "effect"), dim=-1)
    adjacent = (a * b).sum(dim=-1)
    perm = torch.randperm(len(b), generator=torch.Generator().manual_seed(seed + 991))
    shuffled = (a * b[perm]).sum(dim=-1)
    return {
        "adjacent_cosine": float(adjacent.mean()),
        "shuffled_cosine": float(shuffled.mean()),
    }


def _manifest_checks(
    cfg: DictConfig,
    manifest: CacheManifest,
    cte_path: Path,
    stats_path: Path,
    semantic_path: Path,
    payload: dict[str, Any],
) -> None:
    expected = {
        "schema_version": "zeva_fastwam_robotwin_cache_v4",
        "history_semantics": "full_episode_prefix",
        "query_step_unit": "raw_action_step",
        "cte_checkpoint_sha256": checkpoint_sha256(cte_path),
        "dataset_stats_sha256": sha256_file(stats_path),
        "dataset_path": str(cfg.data.train.dataset_dirs[0]),
        "camera_keys": ("cam_high", "cam_left_wrist", "cam_right_wrist"),
        "action_dim": 14,
        "action_group_size": 4,
        "action_horizon": 32,
        "video_frames": 9,
        "phase_dim": 128,
        "effect_dim": 128,
        "image_channels": 48,
        "latent_channels": 48,
        "cte_input_type": "wan_vae_latent",
        "cte_vae_input_size": _video_size_hw(cfg),
        "effect_window_transitions": 4,
        "transition_count": 8,
        "feature_dtype": "float32",
        "action_video_freq_ratio": 4,
        "action_normalization": "fastwam_processor_output",
    }
    manifest.validate(expected)
    checkpoint_vae = dict(payload.get("vae_metadata", {}))
    validate_vae_metadata(checkpoint_vae, manifest.vae_metadata)
    semantic_identity = dict(payload.get("semantic_task_identity", {}))
    if str(semantic_identity.get("sha256", "")) != sha256_file(semantic_path):
        raise ValueError("CTE checkpoint semantic-task SHA does not match current semantic map")


def _build_model_from_checkpoint(cte_path: Path, device: torch.device):
    payload = torch.load(cte_path, map_location="cpu", weights_only=False)
    raw_cfg = dict(payload.get("config", payload.get("model_config", {})))
    cte_cfg = dict(raw_cfg.get("cte", raw_cfg))
    allowed = set(CausalTransitionEncoderConfig.__dataclass_fields__)
    model = CausalTransitionEncoder(
        CausalTransitionEncoderConfig(**{k: v for k, v in cte_cfg.items() if k in allowed})
    )
    load_cte_checkpoint(cte_path, model, map_location="cpu")
    model.to(device).eval().requires_grad_(False)
    if next(model.parameters()).dtype != torch.float32:
        raise RuntimeError("formal CTE must run in float32")
    return model, payload


def _deep_recompute(
    cfg: DictConfig,
    *,
    cache_root: Path,
    rows: list[dict[str, Any]],
    cte_path: Path,
    latent_path: Path,
    stats_path: Path,
    semantic_path: Path,
    deep_episodes: int,
    atol: float,
    rtol: float,
    seed: int,
) -> dict[str, float]:
    if deep_episodes <= 0:
        return {"episodes": 0, "max_abs": 0.0, "mean_abs": 0.0}
    if not torch.cuda.is_available():
        raise RuntimeError("deep cache recomputation requires CUDA")
    device = torch.device("cuda:0")
    model, payload = _build_model_from_checkpoint(cte_path, device)
    latent_cache = CachedCTELatentWindowDataset(
        str(latent_path),
        expected_dataset_stats_sha256=sha256_file(stats_path),
        expected_semantic_task_sha256=sha256_file(semantic_path),
        expected_video_size=_video_size_hw(cfg),
        expected_action_dim=model.cfg.action_dim,
        expected_transition_steps=model.cfg.transition_steps,
        expected_latent_channels=model.cfg.image_channels,
    )
    validate_vae_metadata(dict(payload.get("vae_metadata", {})), latent_cache.vae_metadata)

    with open_dict(cfg.data.train):
        cfg.data.train.video_backend = "pyav"
        cfg.data.train.use_text_embed_cache = False
    BaseLerobotDataset.presample_images = True
    base = instantiate(cfg.data.train)
    dataset = ZevaRobotWinDataset(base)
    plans = _build_episode_plans_metadata_only(
        dataset,
        sample_stride=1,
        transition_steps=model.cfg.transition_steps,
        source_window_actions=model.cfg.transition_steps * 8,
        show_progress=False,
    )
    plans = _canonicalize_plans(plans, latent_cache)
    _validate_plan_alignment(plans, latent_cache)

    # Prefer a mixture of tail and non-tail episodes when both exist.
    rng = random.Random(seed)
    tail_plans = [plan for plan in plans if _tail_required(plan, latent_cache)]
    plain_plans = [plan for plan in plans if not _tail_required(plan, latent_cache)]
    rng.shuffle(tail_plans)
    rng.shuffle(plain_plans)
    selected = []
    half = deep_episodes // 2
    selected.extend(tail_plans[: min(half, len(tail_plans))])
    selected.extend(plain_plans[: min(deep_episodes - len(selected), len(plain_plans))])
    if len(selected) < deep_episodes:
        remaining = [p for p in plans if p not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: deep_episodes - len(selected)])

    selected_ids = {plan.episode_id for plan in selected}
    cache_rows = [row for row in rows if str(row["episode_id"]) in selected_ids]
    # Materialize all selected cache tensors once.
    phase_rows = [row for row in cache_rows if row["record_type"] == "phase_query"]
    effect_rows = [row for row in cache_rows if row["record_type"] == "effect"]
    phase_values = _load_selected_vectors(cache_root, phase_rows, "phase_pre")
    effect_values = _load_selected_vectors(cache_root, effect_rows, "effect")
    cache_lookup: dict[tuple[Any, ...], torch.Tensor] = {}
    for row, value in zip(phase_rows, phase_values, strict=True):
        cache_lookup[(str(row["episode_id"]), "phase_query", int(row["raw_step"]))] = value
    for row, value in zip(effect_rows, effect_values, strict=True):
        cache_lookup[(str(row["episode_id"]), "effect", int(row["start_raw_step"]), int(row["end_raw_step"]))] = value

    need_tail = any(_tail_required(plan, latent_cache) for plan in selected)
    encoder = None
    vae = None
    if need_tail:
        model_values = _cfg_dict(cfg.model)
        vae, vae_metadata = load_frozen_wan_vae(
            model_id=str(model_values.get("model_id", "Wan-AI/Wan2.2-TI2V-5B")),
            tokenizer_model_id=str(model_values.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B")),
            device=str(device),
            torch_dtype=torch.bfloat16,
            redirect_common_files=bool(model_values.get("redirect_common_files", True)),
        )
        validate_vae_metadata(dict(payload.get("vae_metadata", {})), vae_metadata)
        encoder = FastWAMCTELatentEncoder(
            vae,
            resize=_video_size_hw(cfg),
            expected_channels=model.cfg.image_channels,
            input_range="minus_one_one",
        ).encode_history

    abs_errors: list[torch.Tensor] = []
    for pos, plan in enumerate(selected, start=1):
        tail = None
        if _tail_required(plan, latent_cache):
            assert encoder is not None
            item = _TailSourceDataset(dataset, [plan])[0]
            rgb = item["frames"].unsqueeze(0).to(device)
            with torch.inference_mode():
                latent = encoder(rgb)
            if tuple(latent.shape[1:]) != tuple(latent_cache.latent_shape):
                raise RuntimeError(
                    f"deep tail latent shape mismatch for {plan.episode_id}: "
                    f"{tuple(latent.shape[1:])} vs {latent_cache.latent_shape}"
                )
            tail = (
                int(item["episode_step"]),
                latent[0].to(torch.bfloat16).cpu(),
                item["actions"].float().cpu(),
            )
            del rgb, latent

        decoded = _materialize_episode(plan, latent_cache=latent_cache, tail=tail)
        with torch.inference_mode():
            records, _ = _compute_episode(
                decoded,
                model=model,
                cte_device=device,
                cte_input_type="wan_vae_latent",
                frame_encoder=None,
                profile_gpu_timing=False,
                pin_memory=False,
            )
        for record in records:
            if record["record_type"] == "phase_query":
                key = (plan.episode_id, "phase_query", int(record["raw_step"]))
                actual = record["phase_pre"].float()
            else:
                key = (
                    plan.episode_id,
                    "effect",
                    int(record["start_raw_step"]),
                    int(record["end_raw_step"]),
                )
                actual = record["effect"].float()
            expected = cache_lookup.get(key)
            if expected is None:
                raise RuntimeError(f"deep check missing cache key: {key}")
            error = (actual - expected).abs()
            abs_errors.append(error)
            if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
                raise RuntimeError(
                    f"deep recomputation mismatch for {key}: "
                    f"max_abs={float(error.max()):.8g}, mean_abs={float(error.mean()):.8g}, "
                    f"atol={atol}, rtol={rtol}"
                )
        print(f"deep recompute: {pos}/{len(selected)} {plan.episode_id}")

    if vae is not None:
        del vae
    if abs_errors:
        flat = torch.cat([value.flatten() for value in abs_errors])
        max_abs = float(flat.max())
        mean_abs = float(flat.mean())
    else:
        max_abs = mean_abs = 0.0
    return {"episodes": len(selected), "max_abs": max_abs, "mean_abs": mean_abs}


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    validation = _cfg_dict(cfg.get("phase_effect_validation"))
    seed = int(validation.get("seed", 20260924))
    sample_phase = int(validation.get("sample_phase", 4096))
    sample_effect = int(validation.get("sample_effect", 4096))
    pair_samples = int(validation.get("pair_samples", 20000))
    progression_episodes = int(validation.get("progression_episodes", 64))
    adjacent_effect_pairs = int(validation.get("adjacent_effect_pairs", 2048))
    deep_episodes = int(validation.get("deep_episodes", 8))
    deep_atol = float(validation.get("deep_atol", 1.0e-5))
    deep_rtol = float(validation.get("deep_rtol", 1.0e-5))

    zeva = cfg.model.get("zeva", {})
    cte_path = Path(str(zeva.get("cte", {}).get("checkpoint"))).expanduser().resolve()
    latent_path = Path(str(zeva.get("cte", {}).get("latent_cache_path"))).expanduser().resolve()
    cache_root = Path(str(zeva.get("cache", {}).get("path"))).expanduser().resolve()
    stats_path = Path(str(cfg.data.train.get("pretrained_norm_stats", ""))).expanduser().resolve()
    semantic_path = Path(str(cfg.data.train.get("semantic_task_map_path", ""))).expanduser().resolve()
    for path, name in (
        (cte_path, "CTE checkpoint"),
        (stats_path, "dataset stats"),
        (semantic_path, "semantic map"),
        (cache_root / "manifest.json", "cache manifest"),
        (cache_root / "episode_index.json", "cache index"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    if not latent_path.is_dir():
        raise FileNotFoundError(f"latent cache not found: {latent_path}")

    payload = torch.load(cte_path, map_location="cpu", weights_only=False)
    manifest = CacheManifest(**json.loads((cache_root / "manifest.json").read_text()))
    _manifest_checks(cfg, manifest, cte_path, stats_path, semantic_path, payload)

    print("========== phase/effect cache-v4 validation ==========")
    print(f"cache: {cache_root}")
    print(f"CTE sha256: {manifest.cte_checkpoint_sha256}")
    print(f"history_semantics: {manifest.history_semantics}")
    print(f"query_step_unit: {manifest.query_step_unit}")
    print("manifest compatibility: PASSED")

    indexed = _index_cache_rows(cache_root, sample_phase, sample_effect, seed)
    rows: list[dict[str, Any]] = indexed["rows"]
    print(f"records: {len(rows):,}")
    print(f"phase_query: {indexed['type_counts']['phase_query']:,}")
    print(f"effect: {indexed['type_counts']['effect']:,}")
    print(f"episodes: {len(indexed['episode_ids']):,}")
    print(f"semantic tasks: {len(indexed['task_counts'])}")
    if len(indexed["task_counts"]) != 50:
        raise RuntimeError(f"expected 50 semantic tasks, got {len(indexed['task_counts'])}")

    shard_stats = _scan_all_shards(
        cache_root,
        manifest,
        rows,
        indexed["rows_by_shard"],
    )
    if shard_stats["phase_count"] != indexed["type_counts"]["phase_query"]:
        raise RuntimeError("phase count changed between index and shard scan")
    if shard_stats["effect_count"] != indexed["type_counts"]["effect"]:
        raise RuntimeError("effect count changed between index and shard scan")
    print("all shard shapes/dtypes/offsets/finite/valid: PASSED")

    # Cross-check exact expected counts against current dataset metadata.
    with open_dict(cfg.data.train):
        cfg.data.train.video_backend = "pyav"
        cfg.data.train.use_text_embed_cache = False
    BaseLerobotDataset.presample_images = True
    base = instantiate(cfg.data.train)
    dataset = ZevaRobotWinDataset(base)
    latent_cache = CachedCTELatentWindowDataset(
        str(latent_path),
        expected_dataset_stats_sha256=sha256_file(stats_path),
        expected_semantic_task_sha256=sha256_file(semantic_path),
        expected_video_size=_video_size_hw(cfg),
        expected_action_dim=14,
        expected_transition_steps=4,
        expected_latent_channels=48,
    )
    plans = _build_episode_plans_metadata_only(
        dataset,
        sample_stride=1,
        transition_steps=4,
        source_window_actions=32,
        show_progress=False,
    )
    plans = _canonicalize_plans(plans, latent_cache)
    _validate_plan_alignment(plans, latent_cache)
    expected_phase = sum(plan.query_count for plan in plans)
    expected_effect = sum(_expected_effect_count(plan, 4) for plan in plans)
    if indexed["type_counts"]["phase_query"] != expected_phase:
        raise RuntimeError(
            f"phase_query count mismatch: cache={indexed['type_counts']['phase_query']}, expected={expected_phase}"
        )
    if indexed["type_counts"]["effect"] != expected_effect:
        raise RuntimeError(
            f"effect count mismatch: cache={indexed['type_counts']['effect']}, expected={expected_effect}"
        )
    print(f"dataset-derived counts: PASSED (phase={expected_phase:,}, effect={expected_effect:,})")

    phase_rows = indexed["phase_sample"]
    phase = _load_selected_vectors(cache_root, phase_rows, "phase_pre")
    phase_tasks = [str(row["task_id"]) for row in phase_rows]
    phase_episodes = [str(row["episode_id"]) for row in phase_rows]
    nn_top1 = _same_task_nn_top1(phase, phase_tasks, phase_episodes)
    same_cos, cross_cos = _pair_cosine_stats(
        phase, phase_tasks, phase_episodes, pair_samples=pair_samples, seed=seed + 1
    )

    effect_rows = indexed["effect_sample"]
    effect = _load_selected_vectors(cache_root, effect_rows, "effect")
    effect_rank = _effective_rank(effect)

    rng = random.Random(seed + 2)
    episode_pool = sorted(indexed["episode_ids"])
    rng.shuffle(episode_pool)
    progression = _phase_progression_diagnostic(
        cache_root, rows, episode_pool[: min(progression_episodes, len(episode_pool))]
    )
    adjacent = _adjacent_effect_diagnostic(
        cache_root,
        rows,
        indexed["effect_lookup"],
        pairs=adjacent_effect_pairs,
        seed=seed + 3,
    )

    print("-- phase retrieval --")
    print(f"sampled phase queries: {len(phase_rows):,}")
    print(f"same-task NN top1 (different episode): {nn_top1:.4f}")
    print(f"same-task cosine mean: {same_cos:.4f}")
    print(f"cross-task cosine mean: {cross_cos:.4f}")
    print(f"full-cache phase mean per-dim std: {shard_stats['phase_mean_per_dim_std']:.6f}")
    print("-- phase progression --")
    print(f"phase |increment| mean: {progression['increment_abs_mean']:.6f}")
    print(f"phase positive increment fraction: {progression['positive_fraction']:.4f}")
    print(f"phase margin>=0.01 fraction: {progression['margin_fraction']:.4f}")
    print("-- effect --")
    print(f"sampled effects: {len(effect_rows):,}")
    print(f"full-cache effect mean per-dim std: {shard_stats['effect_mean_per_dim_std']:.6f}")
    print(f"effect effective rank: {effect_rank:.3f} / {manifest.effect_dim}")
    print(f"adjacent-effect cosine: {adjacent['adjacent_cosine']:.4f}")
    print(f"shuffled-effect cosine: {adjacent['shuffled_cosine']:.4f}")

    deep = _deep_recompute(
        cfg,
        cache_root=cache_root,
        rows=rows,
        cte_path=cte_path,
        latent_path=latent_path,
        stats_path=stats_path,
        semantic_path=semantic_path,
        deep_episodes=deep_episodes,
        atol=deep_atol,
        rtol=deep_rtol,
        seed=seed + 4,
    )
    if deep_episodes > 0:
        print("-- deep full-prefix recomputation --")
        print(f"episodes checked: {int(deep['episodes'])}")
        print(f"worst max abs: {deep['max_abs']:.8g}")
        print(f"mean abs: {deep['mean_abs']:.8g}")
        print("deep recomputation: PASSED")

    report = {
        "cache_path": str(cache_root),
        "cte_checkpoint": str(cte_path),
        "cte_checkpoint_sha256": manifest.cte_checkpoint_sha256,
        "records": len(rows),
        "phase_query_records": int(indexed["type_counts"]["phase_query"]),
        "effect_records": int(indexed["type_counts"]["effect"]),
        "episodes": len(indexed["episode_ids"]),
        "semantic_tasks": len(indexed["task_counts"]),
        "phase_mean_per_dim_std": shard_stats["phase_mean_per_dim_std"],
        "effect_mean_per_dim_std": shard_stats["effect_mean_per_dim_std"],
        "same_task_nn_top1": nn_top1,
        "same_task_cosine_mean": same_cos,
        "cross_task_cosine_mean": cross_cos,
        "phase_increment_abs_mean": progression["increment_abs_mean"],
        "phase_positive_increment_fraction": progression["positive_fraction"],
        "phase_margin_fraction": progression["margin_fraction"],
        "effect_effective_rank": effect_rank,
        "adjacent_effect_cosine": adjacent["adjacent_cosine"],
        "shuffled_effect_cosine": adjacent["shuffled_cosine"],
        "deep": deep,
        "status": "PASSED",
    }
    report_value = validation.get("report_path")
    report_path = (
        Path(str(report_value)).expanduser().resolve()
        if report_value not in (None, "", "None", "null")
        else cache_root / "validation_report.json"
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"report: {report_path}")
    print("PHASE/EFFECT CACHE V4 VALIDATION: PASSED")


if __name__ == "__main__":
    main()
