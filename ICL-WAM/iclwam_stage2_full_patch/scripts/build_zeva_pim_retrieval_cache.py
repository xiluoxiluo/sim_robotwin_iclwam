"""Prepare exact same-task phase-retrieval inputs for Zeva Stage-2.

The legacy ZevaStage2Dataset performs Python MemoryBank retrieval for every
training sample.  This script materializes that frozen retrieval once, using
GPU matrix multiplication, while preserving the formal Zeva semantics:

* query = current full-prefix phase
* candidates = completed effect windows from the same semantic task
* exclude = current episode
* score = cosine(query_phase, candidate_phase_pre)
* return candidate normalized phase + normalized effect, top-K
* BIT = last completed effects from the current episode, right-aligned

The resulting cache is sharded and memory-mapped; Stage-2 __getitem__ becomes
O(1) plus the ordinary RobotVideoDataset sample decode.
"""

from __future__ import annotations

from collections import defaultdict, deque
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from safetensors.torch import load_file
from torch.nn import functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_zeva_robotwin_cache import (  # noqa: E402
    _barrier,
    _destroy_process_group,
    _init_distributed,
    _resolve_device,
)
from fastwam.zeva.prepared_stage2 import PIM_CACHE_FORMAT  # noqa: E402
from fastwam.zeva.checkpoint import checkpoint_sha256  # noqa: E402
from fastwam.zeva.schemas import CacheManifest, sha256_file  # noqa: E402


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(default if value in (None, "") else value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be boolean")


def _open_memmap(path: Path, dtype, shape):
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _load_phase_cache_manifest(root: Path) -> CacheManifest:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"phase/effect cache manifest missing: {manifest_path}")
    manifest = CacheManifest(**json.loads(manifest_path.read_text(encoding="utf-8")))
    manifest.validate(
        {
            "schema_version": "zeva_fastwam_robotwin_cache_v4",
            "history_semantics": "full_episode_prefix",
            "query_step_unit": "raw_action_step",
            "feature_dtype": "float32",
            "action_dim": 14,
            "action_group_size": 4,
            "action_horizon": 32,
            "video_frames": 9,
            "action_video_freq_ratio": 4,
        }
    )
    return manifest


def _load_phase_cache_metadata(root: Path) -> tuple[CacheManifest, list[dict[str, Any]]]:
    index_path = root / "episode_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"phase/effect cache index missing: {index_path}")
    manifest = _load_phase_cache_manifest(root)
    rows = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("phase/effect cache index is empty")
    return manifest, rows


def _read_row_tensor(
    root: Path,
    row: dict[str, Any],
    current: tuple[str | None, dict[str, torch.Tensor] | None],
) -> tuple[dict[str, torch.Tensor], tuple[str, dict[str, torch.Tensor]]]:
    shard_name = str(row["shard"])
    current_name, tensors = current
    if current_name != shard_name or tensors is None:
        tensors = load_file(str(root / shard_name), device="cpu")
        current = (shard_name, tensors)
    offset = int(row["offset"])
    return {
        "phase_pre": tensors["phase_pre"][offset].float(),
        "effect": tensors["effect"][offset].float(),
        "valid": tensors["valid"][offset].bool(),
    }, current


def _build_staging(
    *,
    root: Path,
    rows: list[dict[str, Any]],
    manifest: CacheManifest,
    staging: Path,
    bit_size: int,
    semantic_sha: str,
) -> dict[str, Any]:
    phase_count = sum(row.get("record_type") == "phase_query" for row in rows)
    effect_count = sum(row.get("record_type") == "effect" for row in rows)
    if phase_count < 1 or effect_count < 1:
        raise ValueError("phase/effect cache contains no query/effect rows")

    episode_map: dict[str, int] = {}
    task_map: dict[str, int] = {}
    episode_ids: list[str] = []
    task_ids: list[str] = []

    q_window = _open_memmap(staging / "q_window_index.npy", np.int64, (phase_count,))
    q_raw = _open_memmap(staging / "q_raw_step.npy", np.int32, (phase_count,))
    q_ep = _open_memmap(staging / "q_episode_index.npy", np.int32, (phase_count,))
    q_task = _open_memmap(staging / "q_task_index.npy", np.int16, (phase_count,))
    q_phase = _open_memmap(staging / "q_phase.npy", np.float32, (phase_count, manifest.phase_dim))
    q_bit = _open_memmap(
        staging / "q_bit_effects.npy",
        np.float32,
        (phase_count, bit_size, manifest.effect_dim),
    )
    q_bit_mask = _open_memmap(staging / "q_bit_mask.npy", np.uint8, (phase_count, bit_size))
    # New mmap files must not rely on filesystem zero-fill semantics.  Queries
    # before the first completed effect need an exact all-zero BIT prefix.
    q_bit[:] = 0.0
    q_bit_mask[:] = 0

    e_phase = _open_memmap(staging / "e_phase.npy", np.float32, (effect_count, manifest.phase_dim))
    e_effect = _open_memmap(staging / "e_effect.npy", np.float32, (effect_count, manifest.effect_dim))
    e_ep = _open_memmap(staging / "e_episode_index.npy", np.int32, (effect_count,))
    e_task = _open_memmap(staging / "e_task_index.npy", np.int16, (effect_count,))

    histories: dict[int, deque[torch.Tensor]] = defaultdict(lambda: deque(maxlen=bit_size))
    current: tuple[str | None, dict[str, torch.Tensor] | None] = (None, None)
    qi = ei = 0
    progress = tqdm(rows, desc="stage phase/effect cache", unit="record", dynamic_ncols=True)
    for row in progress:
        episode_id = str(row["episode_id"])
        task_id = str(row["task_id"])
        if episode_id not in episode_map:
            episode_map[episode_id] = len(episode_ids)
            episode_ids.append(episode_id)
        if task_id not in task_map:
            task_map[task_id] = len(task_ids)
            task_ids.append(task_id)
        ep_index = episode_map[episode_id]
        task_index = task_map[task_id]
        if task_index > np.iinfo(np.int16).max:
            raise ValueError("too many semantic tasks for int16 task index")

        item, current = _read_row_tensor(root, row, current)
        if not bool(item["valid"].item()):
            raise ValueError("formal v4 cache contains invalid record")
        record_type = str(row["record_type"])
        if record_type == "effect":
            phase = item["phase_pre"]
            effect = item["effect"]
            if phase.shape != (manifest.phase_dim,) or effect.shape != (manifest.effect_dim,):
                raise ValueError("effect feature dimension mismatch while staging")
            # Match MemoryBank.add() exactly for retrieval candidates: it
            # stores CPU float32 L2-normalized phase/effect vectors.  BIT history
            # intentionally keeps the direct cache effect, matching the legacy
            # ZevaStage2Dataset path.
            phase_key = F.normalize(phase.float(), dim=0)
            effect_key = F.normalize(effect.float(), dim=0)
            e_phase[ei] = phase_key.numpy()
            e_effect[ei] = effect_key.numpy()
            e_ep[ei] = ep_index
            e_task[ei] = task_index
            histories[ep_index].append(effect.clone())
            ei += 1
        elif record_type == "phase_query":
            if row.get("window_index") is None:
                raise ValueError("phase_query is missing window_index")
            phase = item["phase_pre"]
            q_window[qi] = int(row["window_index"])
            q_raw[qi] = int(row["raw_step"])
            q_ep[qi] = ep_index
            q_task[qi] = task_index
            q_phase[qi] = phase.numpy()
            history = list(histories.get(ep_index, ()))
            if history:
                take = min(bit_size, len(history))
                for j, effect in enumerate(history[-take:], start=bit_size - take):
                    q_bit[qi, j] = effect.numpy()
                    q_bit_mask[qi, j] = 1
            qi += 1
        else:
            raise ValueError(f"unsupported cache record_type {record_type!r}")
    if qi != phase_count or ei != effect_count:
        raise RuntimeError(f"staging counts mismatch: q={qi}/{phase_count}, e={ei}/{effect_count}")
    if len(task_ids) != 50:
        raise RuntimeError(f"formal RoboTwin cache expected 50 semantic tasks, got {len(task_ids)}")

    for array in (q_window, q_raw, q_ep, q_task, q_phase, q_bit, q_bit_mask, e_phase, e_effect, e_ep, e_task):
        array.flush()
    (staging / "episode_ids.json").write_text(json.dumps(episode_ids) + "\n", encoding="utf-8")
    (staging / "task_ids.json").write_text(json.dumps(task_ids) + "\n", encoding="utf-8")
    stage_manifest = {
        "phase_count": phase_count,
        "effect_count": effect_count,
        "episode_count": len(episode_ids),
        "task_count": len(task_ids),
        "phase_dim": manifest.phase_dim,
        "effect_dim": manifest.effect_dim,
        "bit_size": bit_size,
        "cte_checkpoint_sha256": manifest.cte_checkpoint_sha256,
        "dataset_stats_sha256": manifest.dataset_stats_sha256,
        "dataset_path": manifest.dataset_path,
        "semantic_task_map_sha256": semantic_sha,
        "phase_cache_manifest_sha256": sha256_file(root / "manifest.json"),
        "phase_cache_index_sha256": sha256_file(root / "episode_index.json"),
    }
    (staging / "manifest.json").write_text(json.dumps(stage_manifest, indent=2, sort_keys=True) + "\n")
    return stage_manifest


def _query_range(total: int, rank: int, world_size: int) -> tuple[int, int]:
    per = math.ceil(total / world_size)
    start = min(total, rank * per)
    end = min(total, (rank + 1) * per)
    return start, end


def _stable_topk_memorybank(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast top-K with MemoryBank's exact stable-tie semantics.

    Legacy MemoryBank uses ``torch.argsort(..., stable=True)`` over candidates
    in insertion order.  Full stable argsort for every RoboTwin query is
    unnecessarily expensive, so use topk to find the set, then stabilize the
    selected order.  Only rows whose cutoff has more tied candidates than topk
    selected fall back to a full stable CPU argsort.
    """
    if scores.ndim != 2 or not 1 <= int(k) <= scores.shape[1]:
        raise ValueError("stable top-k received invalid score matrix/k")
    raw_scores, raw_indices = torch.topk(
        scores, int(k), dim=-1, largest=True, sorted=False
    )
    cutoff = raw_scores.min(dim=-1).values
    total_cutoff = scores.eq(cutoff[:, None]).sum(dim=-1)
    selected_cutoff = raw_scores.eq(cutoff[:, None]).sum(dim=-1)
    ambiguous = total_cutoff > selected_cutoff

    # Candidate-local index is insertion order because staging uses the
    # phase/effect cache's sequential effect rows.  Sort selected indices by
    # insertion order first, then stable-sort by score descending.
    insertion_order, _ = torch.sort(raw_indices, dim=-1)
    selected_scores = scores.gather(1, insertion_order)
    score_order = torch.argsort(selected_scores, dim=-1, descending=True, stable=True)
    indices = insertion_order.gather(1, score_order)
    values = scores.gather(1, indices)

    if bool(ambiguous.any()):
        # Exact ties at the top-K boundary are rare.  Match the legacy CPU
        # stable argsort exactly for those rows rather than accepting arbitrary
        # CUDA topk tie membership.
        for row in ambiguous.nonzero(as_tuple=False).flatten().tolist():
            stable = torch.argsort(
                scores[row].detach().cpu(), descending=True, stable=True
            )[: int(k)].to(device=scores.device)
            indices[row] = stable
            values[row] = scores[row].gather(0, stable)
    return values, indices


def _prepare_rank(
    *,
    staging: Path,
    shard_dir: Path,
    start: int,
    end: int,
    phase_dim: int,
    effect_dim: int,
    bit_size: int,
    top_k: int,
    device: torch.device,
    query_batch: int,
    rank: int,
) -> None:
    n = end - start
    if n <= 0:
        shard_dir.mkdir(parents=True, exist_ok=True)
        for name, dtype, shape in (
            ("window_index", np.int64, (0,)),
            ("raw_step", np.int32, (0,)),
            ("episode_index", np.int32, (0,)),
            ("task_index", np.int16, (0,)),
            ("phase", np.float32, (0, phase_dim)),
            ("bit_effects", np.float32, (0, bit_size, effect_dim)),
            ("bit_mask", np.uint8, (0, bit_size)),
            ("pim_phases", np.float32, (0, top_k, phase_dim)),
            ("pim_effects", np.float32, (0, top_k, effect_dim)),
            ("pim_mask", np.uint8, (0, top_k)),
            ("pim_scores", np.float32, (0, top_k)),
        ):
            _open_memmap(shard_dir / f"{name}.npy", dtype, shape).flush()
        return

    q_window = np.load(staging / "q_window_index.npy", mmap_mode="r")
    q_raw = np.load(staging / "q_raw_step.npy", mmap_mode="r")
    q_ep = np.load(staging / "q_episode_index.npy", mmap_mode="r")
    q_task = np.load(staging / "q_task_index.npy", mmap_mode="r")
    q_phase = np.load(staging / "q_phase.npy", mmap_mode="r")
    q_bit = np.load(staging / "q_bit_effects.npy", mmap_mode="r")
    q_bit_mask = np.load(staging / "q_bit_mask.npy", mmap_mode="r")
    e_phase = np.load(staging / "e_phase.npy", mmap_mode="r")
    e_effect = np.load(staging / "e_effect.npy", mmap_mode="r")
    e_ep = np.load(staging / "e_episode_index.npy", mmap_mode="r")
    e_task = np.load(staging / "e_task_index.npy", mmap_mode="r")

    out = {
        "window_index": _open_memmap(shard_dir / "window_index.npy", np.int64, (n,)),
        "raw_step": _open_memmap(shard_dir / "raw_step.npy", np.int32, (n,)),
        "episode_index": _open_memmap(shard_dir / "episode_index.npy", np.int32, (n,)),
        "task_index": _open_memmap(shard_dir / "task_index.npy", np.int16, (n,)),
        "phase": _open_memmap(shard_dir / "phase.npy", np.float32, (n, phase_dim)),
        "bit_effects": _open_memmap(shard_dir / "bit_effects.npy", np.float32, (n, bit_size, effect_dim)),
        "bit_mask": _open_memmap(shard_dir / "bit_mask.npy", np.uint8, (n, bit_size)),
        "pim_phases": _open_memmap(shard_dir / "pim_phases.npy", np.float32, (n, top_k, phase_dim)),
        "pim_effects": _open_memmap(shard_dir / "pim_effects.npy", np.float32, (n, top_k, effect_dim)),
        "pim_mask": _open_memmap(shard_dir / "pim_mask.npy", np.uint8, (n, top_k)),
        "pim_scores": _open_memmap(shard_dir / "pim_scores.npy", np.float32, (n, top_k)),
    }
    out["window_index"][:] = q_window[start:end]
    out["raw_step"][:] = q_raw[start:end]
    out["episode_index"][:] = q_ep[start:end]
    out["task_index"][:] = q_task[start:end]
    out["phase"][:] = q_phase[start:end]
    out["bit_effects"][:] = q_bit[start:end]
    out["bit_mask"][:] = q_bit_mask[start:end]
    out["pim_mask"][:] = 0
    out["pim_scores"][:] = -np.inf

    local_tasks = np.unique(np.asarray(q_task[start:end]))
    progress = tqdm(local_tasks.tolist(), desc=f"PIM retrieve rank {rank}", unit="task", position=rank, dynamic_ncols=True)
    for task in progress:
        candidate_idx = np.flatnonzero(np.asarray(e_task) == task)
        local_rel = np.flatnonzero(np.asarray(q_task[start:end]) == task)
        if len(candidate_idx) < top_k:
            raise RuntimeError(f"task {task} has fewer than top_k={top_k} effect candidates")
        cand_phase = torch.from_numpy(np.array(e_phase[candidate_idx], copy=True)).to(device=device, dtype=torch.float32)
        cand_effect = torch.from_numpy(np.array(e_effect[candidate_idx], copy=True)).to(device=device, dtype=torch.float32)
        cand_episode = torch.from_numpy(np.array(e_ep[candidate_idx], copy=True)).to(device=device, dtype=torch.long)
        # Candidates were normalized on CPU during staging to match
        # MemoryBank.add(); transfer them without another normalization.

        for pos in range(0, len(local_rel), query_batch):
            rel = local_rel[pos : pos + query_batch]
            global_idx = start + rel
            query = torch.from_numpy(np.array(q_phase[global_idx], copy=True)).to(device=device, dtype=torch.float32)
            query = F.normalize(query, dim=-1)
            query_episode = torch.from_numpy(np.array(q_ep[global_idx], copy=True)).to(device=device, dtype=torch.long)
            scores = query @ cand_phase.T
            eligible = cand_episode.unsqueeze(0) != query_episode.unsqueeze(1)
            eligible_count = eligible.sum(dim=1)
            if bool((eligible_count < top_k).any()):
                raise RuntimeError(f"task {task} has insufficient leave-one-episode-out candidates")
            scores = scores.masked_fill(~eligible, -torch.inf)
            top_scores, top_local = _stable_topk_memorybank(scores, top_k)
            selected_phase = cand_phase[top_local]
            selected_effect = cand_effect[top_local]
            if not bool(torch.isfinite(top_scores).all()):
                raise FloatingPointError("PIM retrieval produced non-finite selected scores")
            out["pim_phases"][rel] = selected_phase.cpu().numpy()
            out["pim_effects"][rel] = selected_effect.cpu().numpy()
            out["pim_scores"][rel] = top_scores.cpu().numpy()
            out["pim_mask"][rel] = 1
        del cand_phase, cand_effect, cand_episode

    for value in out.values():
        value.flush()


def _validate_random(
    *,
    work_root: Path,
    staging: Path,
    shards: list[dict[str, Any]],
    top_k: int,
    samples: int,
    seed: int,
) -> dict[str, float]:
    q_phase = np.load(staging / "q_phase.npy", mmap_mode="r")
    q_ep = np.load(staging / "q_episode_index.npy", mmap_mode="r")
    q_task = np.load(staging / "q_task_index.npy", mmap_mode="r")
    e_phase = np.load(staging / "e_phase.npy", mmap_mode="r")
    e_effect = np.load(staging / "e_effect.npy", mmap_mode="r")
    e_ep = np.load(staging / "e_episode_index.npy", mmap_mode="r")
    e_task = np.load(staging / "e_task_index.npy", mmap_mode="r")
    total = len(q_phase)
    generator = torch.Generator().manual_seed(seed)
    chosen = torch.randperm(total, generator=generator)[: min(samples, total)].tolist()

    def prepared(global_index: int):
        for shard in shards:
            if shard["global_start"] <= global_index < shard["global_end"]:
                local = global_index - shard["global_start"]
                root = work_root / shard["path"]
                return (
                    np.load(root / "pim_phases.npy", mmap_mode="r")[local],
                    np.load(root / "pim_effects.npy", mmap_mode="r")[local],
                    np.load(root / "pim_scores.npy", mmap_mode="r")[local],
                )
        raise IndexError(global_index)

    worst_phase = worst_effect = worst_score = 0.0
    for index in chosen:
        task = int(q_task[index])
        episode = int(q_ep[index])
        candidate = np.flatnonzero((np.asarray(e_task) == task) & (np.asarray(e_ep) != episode))
        phase = torch.from_numpy(np.array(e_phase[candidate], copy=True)).float()
        effect = torch.from_numpy(np.array(e_effect[candidate], copy=True)).float()
        query = F.normalize(torch.from_numpy(np.array(q_phase[index], copy=True)).float(), dim=-1)
        scores = phase @ query
        order = torch.argsort(scores, descending=True, stable=True)[:top_k]
        expected_phase = phase[order]
        expected_effect = effect[order]
        expected_score = scores[order]
        got_phase, got_effect, got_score = prepared(index)
        got_phase = torch.from_numpy(np.array(got_phase, copy=True)).float()
        got_effect = torch.from_numpy(np.array(got_effect, copy=True)).float()
        got_score = torch.from_numpy(np.array(got_score, copy=True)).float()
        # topk and stable argsort are identical unless an exact cutoff tie occurs.
        # In that exceptionally rare case, require equal scores and vector sets.
        phase_diff = (expected_phase - got_phase).abs().max().item()
        effect_diff = (expected_effect - got_effect).abs().max().item()
        score_diff = (expected_score - got_score).abs().max().item()
        tolerance = 1.0e-5
        if max(phase_diff, effect_diff, score_diff) > tolerance:
            raise RuntimeError(
                f"prepared PIM retrieval mismatch at query {index}: "
                f"phase={phase_diff}, effect={effect_diff}, score={score_diff}"
            )
        worst_phase = max(worst_phase, phase_diff)
        worst_effect = max(worst_effect, effect_diff)
        worst_score = max(worst_score, score_diff)
    return {
        "samples": float(len(chosen)),
        "worst_phase_abs": worst_phase,
        "worst_effect_abs": worst_effect,
        "worst_score_abs": worst_score,
    }


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    info = _init_distributed()
    try:
        zeva = cfg.model.zeva
        phase_root = Path(str(zeva.cache.path)).expanduser().resolve()
        output_root = Path(str(zeva.prepared_stage2.pim_cache_path)).expanduser().resolve()
        semantic_path = Path(str(cfg.data.train.semantic_task_map_path)).expanduser().resolve()
        if not phase_root.is_dir():
            raise FileNotFoundError(phase_root)
        if not semantic_path.is_file():
            raise FileNotFoundError(semantic_path)
        if str(zeva.memory.pim_retrieval_mode) != "phase":
            raise ValueError(
                "formal prepared Stage-2 currently supports pim_retrieval_mode=phase only; "
                "keep cross_task_effect on the legacy diagnostic path"
            )
        top_k = int(zeva.memory.pim_top_k)
        bit_size = int(zeva.memory.bit_size)
        if top_k != int(zeva.prompt.persistent_length):
            raise ValueError("PIM top_k must match prompt persistent_length")
        if bit_size != int(zeva.prompt.brief_length):
            raise ValueError("BIT size must match prompt brief_length")

        overwrite = _env_bool("ZEVA_PIM_CACHE_OVERWRITE", False)
        if info.rank == 0 and output_root.exists() and not overwrite:
            raise FileExistsError(f"prepared PIM cache exists: {output_root}")
        _barrier(info)
        # All ranks parse only the tiny manifest.  The 1.65M-row JSON index is
        # deliberately loaded on rank 0 only; loading it 16 times wastes tens
        # of GB of aggregate host RAM and has no distributed benefit.
        manifest = _load_phase_cache_manifest(phase_root)
        semantic_sha = sha256_file(semantic_path)
        cte_path = Path(str(zeva.cte.checkpoint)).expanduser().resolve()
        if not cte_path.is_file():
            raise FileNotFoundError(f"missing model.zeva.cte.checkpoint: {cte_path}")
        cte_sha = checkpoint_sha256(cte_path)
        if manifest.cte_checkpoint_sha256 != cte_sha:
            raise ValueError("phase/effect cache CTE checkpoint differs from prepared-PIM config")
        cte_payload = torch.load(cte_path, map_location="cpu", weights_only=False)
        cte_identity = cte_payload.get("semantic_task_identity") or {}
        if str(cte_identity.get("sha256")) != semantic_sha:
            raise ValueError("semantic task map differs from the CTE checkpoint used by phase/effect cache")
        if manifest.phase_dim != int(zeva.prompt.phase_dim) or manifest.effect_dim != int(zeva.prompt.effect_dim):
            raise ValueError("phase/effect cache dimensions differ from CausalPrompt config")
        if manifest.dataset_stats_sha256 != sha256_file(str(cfg.data.train.pretrained_norm_stats)):
            raise ValueError("phase/effect cache dataset stats differ from Stage-2 config")
        if manifest.dataset_path != str(cfg.data.train.dataset_dirs[0]):
            raise ValueError("phase/effect cache dataset path differs from Stage-2 config")

        work_root = output_root.parent / f".{output_root.name}.building"
        staging = work_root / "staging"
        if info.rank == 0:
            _manifest_rank0, rows = _load_phase_cache_metadata(phase_root)
            if _manifest_rank0.to_dict() != manifest.to_dict():
                raise RuntimeError("rank-0 phase/effect manifest changed during staging")
            if work_root.exists():
                shutil.rmtree(work_root)
            staging.mkdir(parents=True, exist_ok=True)
            stage_manifest = _build_staging(
                root=phase_root,
                rows=rows,
                manifest=manifest,
                staging=staging,
                bit_size=bit_size,
                semantic_sha=semantic_sha,
            )
            print("========== prepared PIM staging ==========")
            print(json.dumps(stage_manifest, indent=2, sort_keys=True))
        _barrier(info)
        stage_manifest = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
        total = int(stage_manifest["phase_count"])
        start, end = _query_range(total, info.rank, info.world_size)
        shard_dir = work_root / f"shard_{info.rank:03d}"
        query_batch = _env_int("ZEVA_PIM_QUERY_BATCH", 1024)
        device = _resolve_device(cfg, info)
        _prepare_rank(
            staging=staging,
            shard_dir=shard_dir,
            start=start,
            end=end,
            phase_dim=manifest.phase_dim,
            effect_dim=manifest.effect_dim,
            bit_size=bit_size,
            top_k=top_k,
            device=device,
            query_batch=query_batch,
            rank=info.rank,
        )
        print(f"[rank {info.rank}] prepared queries {start:,}:{end:,} ({end-start:,})")
        _barrier(info)

        if info.rank == 0:
            shards = []
            for rank in range(info.world_size):
                s, e = _query_range(total, rank, info.world_size)
                shards.append(
                    {
                        "path": f"shard_{rank:03d}",
                        "global_start": s,
                        "global_end": e,
                    }
                )
            validation = _validate_random(
                work_root=work_root,
                staging=staging,
                shards=shards,
                top_k=top_k,
                samples=_env_int("ZEVA_PIM_VALIDATE", 128),
                seed=int(cfg.get("seed", 42)),
            )
            final_manifest = {
                "format": PIM_CACHE_FORMAT,
                "retrieval_mode": "phase",
                "history_semantics": "full_episode_prefix",
                "query_step_unit": "raw_action_step",
                "num_queries": total,
                "num_effects": int(stage_manifest["effect_count"]),
                "num_episodes": int(stage_manifest["episode_count"]),
                "semantic_task_count": int(stage_manifest["task_count"]),
                "phase_dim": int(manifest.phase_dim),
                "effect_dim": int(manifest.effect_dim),
                "top_k": top_k,
                "bit_size": bit_size,
                "cte_checkpoint_sha256": manifest.cte_checkpoint_sha256,
                "dataset_stats_sha256": manifest.dataset_stats_sha256,
                "dataset_path": manifest.dataset_path,
                "semantic_task_map_sha256": semantic_sha,
                "phase_cache_manifest_sha256": stage_manifest["phase_cache_manifest_sha256"],
                "phase_cache_index_sha256": stage_manifest["phase_cache_index_sha256"],
                "feature_dtype": "float32",
                "shards": shards,
                "validation": validation,
            }
            shutil.copy2(staging / "episode_ids.json", work_root / "episode_ids.json")
            shutil.copy2(staging / "task_ids.json", work_root / "task_ids.json")
            (work_root / "manifest.json").write_text(
                json.dumps(final_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            shutil.rmtree(staging)
            if output_root.exists():
                shutil.rmtree(output_root)
            work_root.rename(output_root)
            print("========== prepared PIM cache ==========")
            print(f"queries: {total:,}")
            print(f"effects: {int(stage_manifest['effect_count']):,}")
            print(f"episodes: {int(stage_manifest['episode_count']):,}")
            print(f"semantic tasks: {int(stage_manifest['task_count'])}")
            print(f"validation: {validation}")
            print(f"output: {output_root}")
            print("PREPARED PIM CACHE: PASSED")
        _barrier(info)
    finally:
        _destroy_process_group(info)


if __name__ == "__main__":
    main()
