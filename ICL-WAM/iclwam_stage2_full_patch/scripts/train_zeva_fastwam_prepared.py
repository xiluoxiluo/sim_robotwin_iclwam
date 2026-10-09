"""Formal Stage-2 Zeva/FastWAM training from prepared offline conditioning.

This entrypoint preserves the model/trainer semantics of train_zeva_fastwam.py
but replaces expensive per-sample Python PIM retrieval and per-startup static
retrieval with immutable, identity-checked prepared artifacts.
"""

from __future__ import annotations

from pathlib import Path
import os
import json

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from fastwam.runtime import _mixed_precision_to_model_dtype
from fastwam.trainer import Wan22Trainer
from fastwam.zeva.checkpoint import checkpoint_sha256
from fastwam.zeva.prepared_stage2 import (
    PreparedPIMCache,
    PreparedZevaStage2Dataset,
    StaticEpisodeContextCache,
    validate_static_artifact_hashes,
)
from fastwam.zeva.schemas import sha256_file
from fastwam.zeva.static_task_context import task_context_identity


def _require_file(value, name: str) -> Path:
    if value in (None, "", "None", "null"):
        raise ValueError(f"Set {name}")
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {name}: {path}")
    return path


def _resolve_process_device() -> torch.device:
    """Bind torchrun/Accelerate workers before a multi-billion-param model is built."""
    if not torch.cuda.is_available():
        raise RuntimeError("Zeva Stage-2 requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    count = torch.cuda.device_count()
    if local_rank < 0 or local_rank >= count:
        raise ValueError(f"LOCAL_RANK={local_rank} outside visible CUDA device count={count}")
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Zeva Stage-2 requires CUDA")
    device = _resolve_process_device()
    if not bool(cfg.data.train.get("use_text_embed_cache", False)):
        raise ValueError("Stage-2 requires data.train.use_text_embed_cache=true")
    if int(cfg.data.train.get("global_sample_stride", -1)) != 1:
        raise ValueError("Stage-2 requires global_sample_stride=1")
    if int(cfg.data.train.get("action_video_freq_ratio", -1)) != 4:
        raise ValueError("Stage-2 requires action_video_freq_ratio=4")

    zeva = cfg.model.zeva
    if str(zeva.task_context.mode) != "static":
        raise ValueError("formal prepared Stage-2 requires task_context.mode=static")
    if str(zeva.memory.pim_retrieval_mode) != "phase":
        raise ValueError("formal prepared Stage-2 requires Zeva default pim_retrieval_mode=phase")

    base_ckpt = _require_file(cfg.get("ckpt"), "ckpt")
    cte_ckpt = _require_file(zeva.cte.checkpoint, "model.zeva.cte.checkpoint")
    bank_path = _require_file(zeva.task_context.bank_path, "model.zeva.task_context.bank_path")
    readout_path = _require_file(
        zeva.task_context.readout_cache_path,
        "model.zeva.task_context.readout_cache_path",
    )
    head_path = _require_file(
        zeva.task_context.retrieval_checkpoint,
        "model.zeva.task_context.retrieval_checkpoint",
    )
    static_path = _require_file(
        zeva.task_context.static_episode_cache_path,
        "model.zeva.task_context.static_episode_cache_path",
    )
    pim_root = Path(str(zeva.prepared_stage2.pim_cache_path)).expanduser().resolve()
    if not pim_root.is_dir():
        raise FileNotFoundError(f"missing model.zeva.prepared_stage2.pim_cache_path: {pim_root}")
    phase_root = Path(str(zeva.cache.path)).expanduser().resolve()
    if not phase_root.is_dir():
        raise FileNotFoundError(f"missing validated model.zeva.cache.path: {phase_root}")
    phase_manifest_path = phase_root / "manifest.json"
    phase_index_path = phase_root / "episode_index.json"
    if not phase_manifest_path.is_file() or not phase_index_path.is_file():
        raise FileNotFoundError(f"incomplete phase/effect cache: {phase_root}")

    stats_path = _require_file(cfg.data.train.pretrained_norm_stats, "data.train.pretrained_norm_stats")
    semantic_path = _require_file(cfg.data.train.semantic_task_map_path, "data.train.semantic_task_map_path")
    base_sha = checkpoint_sha256(base_ckpt)
    cte_sha = checkpoint_sha256(cte_ckpt)
    stats_sha = sha256_file(stats_path)
    semantic_sha = sha256_file(semantic_path)
    top_k = int(zeva.memory.pim_top_k)
    bit_size = int(zeva.memory.bit_size)
    if top_k != int(zeva.prompt.persistent_length):
        raise ValueError("PIM top_k must match prompt persistent_length")
    if bit_size != int(zeva.prompt.brief_length):
        raise ValueError("BIT size must match prompt brief_length")

    cte_payload = torch.load(cte_ckpt, map_location="cpu", weights_only=False)
    cte_identity = cte_payload.get("semantic_task_identity") or {}
    if str(cte_identity.get("sha256")) != semantic_sha:
        raise ValueError("Stage-2 semantic map differs from CTE checkpoint")
    video_size = [int(v) for v in cfg.data.train.video_size]
    if list(cte_payload.get("cte_vae_input_size", ())) != video_size:
        raise ValueError("Stage-2 video size differs from CTE VAE input size")
    if tuple(cte_payload.get("camera_keys", ())) != ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        raise ValueError("Stage-2 camera order differs from CTE checkpoint")

    pim = PreparedPIMCache(
        pim_root,
        expected={
            "retrieval_mode": "phase",
            "history_semantics": "full_episode_prefix",
            "query_step_unit": "raw_action_step",
            "phase_dim": int(zeva.prompt.phase_dim),
            "effect_dim": int(zeva.prompt.effect_dim),
            "top_k": top_k,
            "bit_size": bit_size,
            "cte_checkpoint_sha256": cte_sha,
            "dataset_stats_sha256": stats_sha,
            "dataset_path": str(cfg.data.train.dataset_dirs[0]),
            "semantic_task_map_sha256": semantic_sha,
            "phase_cache_manifest_sha256": sha256_file(phase_manifest_path),
            "phase_cache_index_sha256": sha256_file(phase_index_path),
            "feature_dtype": "float32",
        },
    )
    static_context = StaticEpisodeContextCache(static_path)
    validate_static_artifact_hashes(
        static_context,
        bank_path=bank_path,
        head_path=head_path,
        readout_path=readout_path,
    )
    static_expected = {
        "base_checkpoint_sha256": base_sha,
        "cte_checkpoint_sha256": cte_sha,
        "dataset_stats_sha256": stats_sha,
        "semantic_task_map_sha256": semantic_sha,
        "video_size": video_size,
        "context_len": int(cfg.model.tokenizer_max_len),
    }
    for key, expected in static_expected.items():
        if static_context.metadata.get(key) != expected:
            raise ValueError(
                f"static context source identity mismatch for {key}: "
                f"expected={expected!r}, got={static_context.metadata.get(key)!r}"
            )
    if int(static_context.metadata.get("top_k", -1)) != int(zeva.task_context.top_k):
        raise ValueError("static context top_k differs from Stage-2 task_context.top_k")
    if static_context.value_dim != int(zeva.prompt.global_dim):
        raise ValueError("static context value dim differs from prompt.global_dim")

    base = instantiate(cfg.data.train)
    if getattr(base, "semantic_task_identity", None) is None:
        raise ValueError("Stage-2 dataset did not load semantic_task_map_path")
    if str(base.semantic_task_identity.get("sha256")) != semantic_sha:
        raise ValueError("Stage-2 dataset semantic identity mismatch")
    dataset = PreparedZevaStage2Dataset(base, pim, static_context)

    model = instantiate(
        cfg.model,
        model_dtype=_mixed_precision_to_model_dtype(str(cfg.mixed_precision)),
        device=str(device),
    )
    model.load_checkpoint(str(base_ckpt))
    model.zeva_task_context_identity = task_context_identity(
        "static",
        str(bank_path),
        str(head_path),
        int(zeva.task_context.top_k),
    )

    print("========== prepared Stage-2 ==========")
    print(f"training_stage: {zeva.training_stage}")
    print(f"dataset queries: {len(dataset):,}")
    print(f"prepared PIM: {pim_root}")
    print(f"static context: {static_path}")
    print(f"base sha256: {base_sha}")
    print(f"CTE sha256: {cte_sha}")
    print(f"semantic map sha256: {semantic_sha}")
    print("PREPARED STAGE-2 PREFLIGHT: PASSED")

    trainer = Wan22Trainer(cfg=cfg, model=model, train_dataset=dataset, val_dataset=None)
    trainer.train()
    print(f"Stage-2 complete; addon checkpoints are under {Path(str(cfg.output_dir)) / 'checkpoints'}")


if __name__ == "__main__":
    main()
