"""Post-Stage-1 readiness checks for RoboTwin Zeva CTE.

Checks:
- checkpoint/cache identity and finite outputs;
- same-task retrieval structure;
- phase progression / non-collapse diagnostics;
- effect non-collapse and paired-vs-shuffled alignment;
- local-window versus reconstructed full-prefix representation drift.

The local/full comparison is diagnostic by default because the Stage-1 model is
trained on reset 32-action windows while cache-v4 intentionally runs a full
episode prefix. Use ``+cte_readiness.strict_local_full=true`` only after you
have chosen a project-specific acceptance threshold.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from fastwam.zeva.causal_transition_encoder import (
    CausalTransitionEncoder,
    CausalTransitionEncoderConfig,
)
from fastwam.zeva.checkpoint import checkpoint_sha256, load_cte_checkpoint
from fastwam.zeva.cte_latent_cache import CachedCTELatentWindowDataset
from fastwam.zeva.schemas import sha256_file
from fastwam.zeva.vae_adapter import validate_vae_metadata


def _cfg_dict(value: Any) -> dict[str, Any]:
    return {} if value is None else dict(OmegaConf.to_container(value, resolve=True))


def _video_size_hw(cfg: DictConfig) -> tuple[int, int]:
    value = cfg.data.train.get("video_size")
    if value is None or len(value) != 2:
        raise ValueError("data.train.video_size must be [H, W]")
    return tuple(int(v) for v in value)


def _load_model(cte_path: Path, device: torch.device) -> tuple[CausalTransitionEncoder, dict[str, Any]]:
    payload = torch.load(cte_path, map_location="cpu", weights_only=False)
    raw_cfg = dict(payload.get("config", payload.get("model_config", {})))
    cte_cfg = dict(raw_cfg.get("cte", raw_cfg))
    allowed = set(CausalTransitionEncoderConfig.__dataclass_fields__)
    model = CausalTransitionEncoder(
        CausalTransitionEncoderConfig(**{k: v for k, v in cte_cfg.items() if k in allowed})
    )
    load_cte_checkpoint(cte_path, model, map_location="cpu")
    model.to(device).eval().requires_grad_(False)
    return model, payload


def _masked_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weight = valid.unsqueeze(-1).to(x.dtype)
    return (x * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


def _quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(torch.quantile(values.float(), torch.tensor(float(q))).item())


def _effective_rank(code: torch.Tensor) -> float:
    if code.ndim != 2 or min(code.shape) < 2:
        return 0.0
    centered = code.float() - code.float().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    total = energy.sum()
    if not torch.isfinite(total) or float(total) <= 0.0:
        return 0.0
    probability = (energy / total).clamp_min(1e-12)
    entropy = -(probability * probability.log()).sum()
    return float(torch.exp(entropy).item())


def _balanced_rows(
    cache: CachedCTELatentWindowDataset,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cache.rows:
        by_task[str(row["task_id"])].append(row)
    rng = random.Random(seed)
    for rows in by_task.values():
        rng.shuffle(rows)
    tasks = sorted(by_task)
    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < min(count, len(cache.rows)):
        made_progress = False
        for task in tasks:
            rows = by_task[task]
            if cursor < len(rows):
                selected.append(rows[cursor])
                made_progress = True
                if len(selected) >= count:
                    break
        if not made_progress:
            break
        cursor += 1
    return selected


def _local_batch(
    rows: list[dict[str, Any]],
    *,
    cache: CachedCTELatentWindowDataset,
    model: CausalTransitionEncoder,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    samples = [cache.get_row(row) for row in rows]
    frames = torch.stack([sample["cte_frames"] for sample in samples]).float().to(device)
    actions = torch.stack([sample["transition_actions"] for sample in samples]).float().to(device)
    frame_valid = torch.stack([sample["frame_valid"] for sample in samples]).bool().to(device)
    transition_valid = torch.stack([sample["transition_valid"] for sample in samples]).bool().to(device)
    return model(
        frames,
        actions,
        valid_mask=frame_valid,
        transition_valid=transition_valid,
    )


def _insert_consistent(
    mapping: dict[int, torch.Tensor],
    raw_step: int,
    value: torch.Tensor,
    *,
    episode_id: str,
    kind: str,
) -> None:
    value = value.detach().cpu().clone()
    previous = mapping.get(int(raw_step))
    if previous is None:
        mapping[int(raw_step)] = value
        return
    if previous.dtype == torch.bfloat16 or value.dtype == torch.bfloat16:
        equal = torch.equal(previous.to(torch.bfloat16), value.to(torch.bfloat16))
    else:
        equal = torch.allclose(previous, value, atol=1e-6, rtol=0.0)
    if not equal:
        raise ValueError(
            f"inconsistent cached {kind}: episode={episode_id}, raw_step={raw_step}"
        )


def _reconstruct_episode(
    cache: CachedCTELatentWindowDataset,
    episode_id: str,
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    rows = cache.rows_by_episode.get(str(episode_id), [])
    if not rows:
        raise KeyError(f"latent cache has no rows for episode {episode_id}")
    boundaries: dict[int, torch.Tensor] = {}
    actions: dict[int, torch.Tensor] = {}
    for row in rows:
        sample = cache.get_row(row)
        start = int(row["episode_step"])
        for local, frame in enumerate(sample["cte_frames"]):
            _insert_consistent(
                boundaries,
                start + 4 * local,
                frame,
                episode_id=episode_id,
                kind="latent boundary",
            )
        for local, action in enumerate(sample["transition_actions"]):
            _insert_consistent(
                actions,
                start + 4 * local,
                action,
                episode_id=episode_id,
                kind="action group",
            )

    raw_steps = sorted(boundaries)
    if not raw_steps or raw_steps[0] != 0:
        raise ValueError(f"episode {episode_id} cache does not start at raw_step=0")
    expected = list(range(raw_steps[0], raw_steps[-1] + 1, 4))
    if raw_steps != expected:
        missing = sorted(set(expected) - set(raw_steps))
        raise ValueError(
            f"episode {episode_id} latent cache is not contiguous; missing={missing[:8]}"
        )
    for step in raw_steps[:-1]:
        if step not in actions:
            raise ValueError(f"episode {episode_id} missing cached action at raw_step={step}")

    frame_tensor = torch.stack([boundaries[step] for step in raw_steps]).float()
    action_tensor = torch.stack([actions[step] for step in raw_steps[:-1]]).float()
    return raw_steps, frame_tensor, action_tensor


def _finite_or_raise(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN/Inf")


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CTE readiness validation requires CUDA")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    zeva = cfg.model.get("zeva", {})
    cte_cfg = _cfg_dict(zeva.get("cte"))
    ready_cfg = _cfg_dict(cfg.get("cte_readiness"))

    cte_value = ready_cfg.get("checkpoint", cte_cfg.get("checkpoint"))
    if cte_value in (None, "", "None", "null"):
        raise ValueError("set model.zeva.cte.checkpoint or +cte_readiness.checkpoint")
    cte_path = Path(str(cte_value)).expanduser().resolve()
    if not cte_path.is_file():
        raise FileNotFoundError(cte_path)

    latent_value = ready_cfg.get("latent_cache_path", cte_cfg.get("latent_cache_path"))
    if latent_value in (None, "", "None", "null"):
        raise ValueError("set model.zeva.cte.latent_cache_path or +cte_readiness.latent_cache_path")

    stats_path = Path(str(cfg.data.train.get("pretrained_norm_stats", ""))).expanduser()
    semantic_path = Path(str(cfg.data.train.get("semantic_task_map_path", ""))).expanduser()
    if not stats_path.is_file() or not semantic_path.is_file():
        raise FileNotFoundError("readiness validation requires stats and semantic task map")

    model, payload = _load_model(cte_path, device)
    if str(payload.get("cte_input_type", cte_cfg.get("input_type", ""))) != "wan_vae_latent":
        raise ValueError("this readiness validator expects wan_vae_latent CTE")

    cache = CachedCTELatentWindowDataset(
        str(latent_value),
        expected_dataset_stats_sha256=sha256_file(stats_path),
        expected_semantic_task_sha256=sha256_file(semantic_path),
        expected_video_size=_video_size_hw(cfg),
        expected_action_dim=model.cfg.action_dim,
        expected_transition_steps=model.cfg.transition_steps,
        expected_latent_channels=model.cfg.image_channels,
    )
    validate_vae_metadata(dict(payload.get("vae_metadata", {})), cache.vae_metadata)

    sample_windows = min(int(ready_cfg.get("sample_windows", 2048)), len(cache.rows))
    batch_size = max(1, int(ready_cfg.get("batch_size", 64)))
    seed = int(cfg.get("seed", 42))
    rows = _balanced_rows(cache, sample_windows, seed)

    retrieval_chunks: list[torch.Tensor] = []
    phase_chunks: list[torch.Tensor] = []
    phase_increment_chunks: list[torch.Tensor] = []
    effect_raw_chunks: list[torch.Tensor] = []
    effect_pre_chunks: list[torch.Tensor] = []
    effect_post_chunks: list[torch.Tensor] = []
    task_ids: list[str] = []
    episode_ids: list[str] = []

    with torch.inference_mode():
        for start in tqdm(range(0, len(rows), batch_size), desc="CTE local readiness", dynamic_ncols=True):
            batch_rows = rows[start : start + batch_size]
            samples = [cache.get_row(row) for row in batch_rows]
            frames = torch.stack([sample["cte_frames"] for sample in samples]).float().to(device)
            actions = torch.stack([sample["transition_actions"] for sample in samples]).float().to(device)
            valid = torch.stack([sample["frame_valid"] for sample in samples]).bool().to(device)
            transition_valid = torch.stack([sample["transition_valid"] for sample in samples]).bool().to(device)
            output = model(frames, actions, valid_mask=valid, transition_valid=transition_valid)

            for key in ("retrieval", "phase", "effect_pre", "effect_post", "effect_post_raw"):
                _finite_or_raise(key, output[key])

            pooled = torch.nn.functional.normalize(_masked_mean(output["retrieval"], valid), dim=-1)
            retrieval_chunks.append(pooled.cpu())
            task_ids.extend(str(row["task_id"]) for row in batch_rows)
            episode_ids.extend(str(row["episode_id"]) for row in batch_rows)

            phase = output["phase"]
            for b in range(phase.shape[0]):
                z = phase[b][valid[b]]
                if z.numel():
                    phase_chunks.append(z.cpu())
                if z.shape[0] >= 3:
                    direction = torch.nn.functional.normalize(z[-1] - z[0], dim=-1)
                    increments = (z[1:] - z[:-1]) @ direction
                    phase_increment_chunks.append(increments.cpu())

            mask = output["effect_complete"]
            if bool(mask.any()):
                effect_raw_chunks.append(output["effect_post_raw"][mask].cpu())
                effect_pre_chunks.append(output["effect_pre"][mask].cpu())
                effect_post_chunks.append(output["effect_post"][mask].cpu())

    retrieval = torch.cat(retrieval_chunks, dim=0)
    phase_tokens = torch.cat(phase_chunks, dim=0)
    phase_increments = torch.cat(phase_increment_chunks, dim=0) if phase_increment_chunks else torch.empty(0)
    effect_raw = torch.cat(effect_raw_chunks, dim=0) if effect_raw_chunks else torch.empty((0, model.cfg.effect_dim))
    effect_pre = torch.cat(effect_pre_chunks, dim=0) if effect_pre_chunks else torch.empty((0, model.cfg.effect_dim))
    effect_post = torch.cat(effect_post_chunks, dim=0) if effect_post_chunks else torch.empty((0, model.cfg.effect_dim))

    # ------------------------------------------------------------------
    # Same-task nearest-neighbour diagnostics.
    # ------------------------------------------------------------------
    sim = retrieval @ retrieval.T
    n = sim.shape[0]
    task_lookup = {name: i for i, name in enumerate(sorted(set(task_ids)))}
    episode_lookup = {name: i for i, name in enumerate(sorted(set(episode_ids)))}
    task_tensor = torch.tensor([task_lookup[name] for name in task_ids], dtype=torch.long)
    episode_tensor = torch.tensor([episode_lookup[name] for name in episode_ids], dtype=torch.long)
    task_equal = task_tensor[:, None].eq(task_tensor[None, :])
    same_episode = episode_tensor[:, None].eq(episode_tensor[None, :])
    candidate = ~same_episode
    candidate.fill_diagonal_(False)
    usable = candidate.any(dim=1)
    masked_sim = sim.masked_fill(~candidate, -torch.inf)
    nearest = masked_sim.argmax(dim=1)
    top1_same_task = float(task_equal[torch.arange(n), nearest][usable].float().mean().item())

    same_mask = task_equal & candidate
    cross_mask = (~task_equal) & candidate
    same_cos = float(sim[same_mask].mean().item()) if bool(same_mask.any()) else float("nan")
    cross_cos = float(sim[cross_mask].mean().item()) if bool(cross_mask.any()) else float("nan")

    retrieval_std = float(retrieval.std(dim=0, correction=0).mean().item())
    phase_std = float(phase_tokens.std(dim=0, correction=0).mean().item())
    phase_motion = (
        float((phase_increments.abs()).mean().item()) if phase_increments.numel() else 0.0
    )
    phase_positive_fraction = (
        float((phase_increments > 0).float().mean().item()) if phase_increments.numel() else 0.0
    )
    phase_margin_fraction = (
        float((phase_increments >= 0.01).float().mean().item()) if phase_increments.numel() else 0.0
    )

    effect_std = float(effect_raw.std(dim=0, correction=0).mean().item()) if effect_raw.numel() else 0.0
    effect_rank = _effective_rank(effect_raw)
    paired_effect_cos = float((effect_pre * effect_post).sum(dim=-1).mean().item()) if effect_pre.numel() else float("nan")
    if effect_post.shape[0] > 1:
        shuffled_effect_cos = float((effect_pre * effect_post.roll(1, dims=0)).sum(dim=-1).mean().item())
    else:
        shuffled_effect_cos = float("nan")

    # ------------------------------------------------------------------
    # Local-window vs full-prefix drift.
    # ------------------------------------------------------------------
    full_episode_count = max(1, int(ready_cfg.get("full_prefix_episodes", 32)))
    windows_per_episode = max(1, int(ready_cfg.get("windows_per_episode", 4)))
    rng = random.Random(seed + 17)
    episode_pool = list(cache.rows_by_episode)
    rng.shuffle(episode_pool)
    selected_episodes = episode_pool[: min(full_episode_count, len(episode_pool))]
    phase_cosines: list[torch.Tensor] = []
    retrieval_cosines: list[torch.Tensor] = []

    with torch.inference_mode():
        for episode_id in tqdm(selected_episodes, desc="CTE full-prefix drift", dynamic_ncols=True):
            raw_steps, frames_cpu, actions_cpu = _reconstruct_episode(cache, episode_id)
            frames = frames_cpu.unsqueeze(0).to(device)
            actions = actions_cpu.unsqueeze(0).to(device)
            full = model(
                frames,
                actions,
                valid_mask=torch.ones((1, len(raw_steps)), dtype=torch.bool, device=device),
                transition_valid=torch.ones(
                    (1, len(raw_steps) - 1, model.cfg.transition_steps),
                    dtype=torch.bool,
                    device=device,
                ),
            )
            full_phase = full["phase"][0]
            full_retrieval = full["retrieval"][0]
            raw_to_index = {raw: i for i, raw in enumerate(raw_steps)}

            local_rows = list(cache.rows_by_episode[episode_id])
            rng.shuffle(local_rows)
            local_rows = local_rows[: min(windows_per_episode, len(local_rows))]
            for row in local_rows:
                sample = cache.get_row(row)
                local = model(
                    sample["cte_frames"].float().unsqueeze(0).to(device),
                    sample["transition_actions"].float().unsqueeze(0).to(device),
                    valid_mask=sample["frame_valid"].bool().unsqueeze(0).to(device),
                    transition_valid=sample["transition_valid"].bool().unsqueeze(0).to(device),
                )
                start_raw = int(row["episode_step"])
                for local_idx in range(local["phase"].shape[1]):
                    raw = start_raw + 4 * local_idx
                    full_idx = raw_to_index.get(raw)
                    if full_idx is None:
                        continue
                    phase_cosines.append(
                        torch.nn.functional.cosine_similarity(
                            local["phase"][0, local_idx], full_phase[full_idx], dim=0
                        ).cpu()
                    )
                    retrieval_cosines.append(
                        torch.nn.functional.cosine_similarity(
                            local["retrieval"][0, local_idx], full_retrieval[full_idx], dim=0
                        ).cpu()
                    )

    phase_local_full = torch.stack(phase_cosines) if phase_cosines else torch.empty(0)
    retrieval_local_full = torch.stack(retrieval_cosines) if retrieval_cosines else torch.empty(0)

    # Collapse checks are deliberately conservative. Retrieval quality and
    # local/full drift are reported rather than silently converted into a
    # project-independent pass/fail threshold.
    hard_failures: list[str] = []
    if retrieval_std <= float(ready_cfg.get("min_retrieval_std", 1e-4)):
        hard_failures.append(f"retrieval appears collapsed (mean dim std={retrieval_std:.6g})")
    if phase_std <= float(ready_cfg.get("min_phase_std", 1e-4)):
        hard_failures.append(f"phase appears collapsed (mean dim std={phase_std:.6g})")
    if effect_raw.numel() == 0:
        hard_failures.append("no complete effect windows were produced")
    elif effect_std <= float(ready_cfg.get("min_effect_std", 1e-4)):
        hard_failures.append(f"effect appears collapsed (mean dim std={effect_std:.6g})")
    if effect_raw.numel() and effect_rank < float(ready_cfg.get("min_effect_rank", 2.0)):
        hard_failures.append(f"effect effective rank is too small ({effect_rank:.3f})")

    strict_local_full = bool(ready_cfg.get("strict_local_full", False))
    min_local_full = float(ready_cfg.get("min_local_full_cosine", 0.50))
    local_full_mean = float(phase_local_full.mean().item()) if phase_local_full.numel() else float("nan")
    if strict_local_full and (not math.isfinite(local_full_mean) or local_full_mean < min_local_full):
        hard_failures.append(
            f"full-prefix/local phase cosine {local_full_mean:.4f} < {min_local_full:.4f}"
        )

    print("========== CTE readiness ==========")
    print(f"checkpoint: {cte_path}")
    print(f"checkpoint sha256: {checkpoint_sha256(cte_path)}")
    print(f"latent cache: {cache.root}")
    print(f"sampled windows: {len(rows):,}")
    print(f"semantic tasks sampled: {len(set(task_ids))}")
    print("-- retrieval --")
    print(f"same-task NN top1 (different episode): {top1_same_task:.4f}")
    print(f"same-task cosine mean: {same_cos:.4f}")
    print(f"cross-task cosine mean: {cross_cos:.4f}")
    print(f"retrieval mean per-dim std: {retrieval_std:.6f}")
    print("-- phase --")
    print(f"phase mean per-dim std: {phase_std:.6f}")
    print(f"phase |increment| mean: {phase_motion:.6f}")
    print(f"phase positive increment fraction: {phase_positive_fraction:.4f}")
    print(f"phase margin>=0.01 fraction: {phase_margin_fraction:.4f}")
    print("-- effect --")
    print(f"effect samples: {effect_raw.shape[0]:,}")
    print(f"effect raw mean per-dim std: {effect_std:.6f}")
    print(f"effect effective rank: {effect_rank:.3f} / {model.cfg.effect_dim}")
    print(f"effect paired cosine: {paired_effect_cos:.4f}")
    print(f"effect shuffled cosine: {shuffled_effect_cos:.4f}")
    print("-- local vs full-prefix (diagnostic) --")
    print(f"episodes checked: {len(selected_episodes)}")
    if phase_local_full.numel():
        print(f"phase cosine mean/p05: {float(phase_local_full.mean()):.4f} / {_quantile(phase_local_full, 0.05):.4f}")
        print(
            "retrieval cosine mean/p05: "
            f"{float(retrieval_local_full.mean()):.4f} / {_quantile(retrieval_local_full, 0.05):.4f}"
        )
    else:
        print("no local/full comparison pairs were available")

    if hard_failures:
        print("CTE READINESS: FAILED")
        for failure in hard_failures:
            print(f"  - {failure}")
        raise RuntimeError("CTE readiness checks failed")

    if (
        phase_local_full.numel()
        and float(phase_local_full.mean()) < min_local_full
        and not strict_local_full
    ):
        print(
            "WARNING: local/full phase drift is large. This is diagnostic-only "
            "because Stage-1 resets its recurrent state while cache-v4 uses a full prefix."
        )
    print("CTE READINESS STRUCTURAL CHECKS: PASSED")


if __name__ == "__main__":
    main()