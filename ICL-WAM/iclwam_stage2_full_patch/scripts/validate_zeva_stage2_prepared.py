"""Validate all prepared Zeva Stage-2 artifacts before expensive training.

The validator is intentionally independent of training state.  It verifies:
* immutable artifact hashes and source-cache identities;
* prepared PIM/static-context dimensions, dtypes, finiteness and coverage;
* exact RobotVideoDataset window/episode/task/raw-step alignment on sampled rows;
* Stage-2 model input shapes;
* optionally, one real frozen-FastWAM + Zeva-addon forward pass.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from fastwam.runtime import _mixed_precision_to_model_dtype
from fastwam.zeva.checkpoint import checkpoint_sha256
from fastwam.zeva.prepared_stage2 import (
    PreparedPIMCache,
    PreparedZevaStage2Dataset,
    StaticEpisodeContextCache,
    validate_static_artifact_hashes,
)
from fastwam.zeva.schemas import sha256_file
from fastwam.zeva.static_task_context import task_context_identity


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be boolean, got {raw!r}")


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw in (None, "") else int(raw)


def _require_file(value, name: str) -> Path:
    if value in (None, "", "None", "null"):
        raise ValueError(f"Set {name}")
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {name}: {path}")
    return path


def _sample_indices(total: int, count: int, seed: int) -> list[int]:
    count = min(max(1, count), total)
    # Combine deterministic edge points with a seeded random interior sample.
    fixed = {0, total - 1, total // 2}
    g = torch.Generator().manual_seed(seed)
    need = max(0, count - len(fixed))
    if need:
        fixed.update(torch.randperm(total, generator=g)[:need].tolist())
    return sorted(fixed)[:count]


def _assert_tensor(name: str, value: torch.Tensor, shape: tuple[int, ...], *, dtype=None) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be Tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} shape mismatch: {tuple(value.shape)} != {shape}")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name} dtype mismatch: {value.dtype} != {dtype}")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains NaN/Inf")


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    zeva = cfg.model.zeva
    if str(zeva.task_context.mode) != "static":
        raise ValueError("prepared Stage-2 validation requires task_context.mode=static")
    if str(zeva.memory.pim_retrieval_mode) != "phase":
        raise ValueError("prepared Stage-2 validation requires pim_retrieval_mode=phase")
    if not bool(cfg.data.train.get("use_text_embed_cache", False)):
        raise ValueError("prepared Stage-2 requires data.train.use_text_embed_cache=true")
    if int(cfg.data.train.global_sample_stride) != 1 or int(cfg.data.train.action_video_freq_ratio) != 4:
        raise ValueError("prepared Stage-2 requires stride=1 and action_video_freq_ratio=4")

    base_ckpt = _require_file(cfg.get("ckpt"), "ckpt")
    cte_ckpt = _require_file(zeva.cte.checkpoint, "model.zeva.cte.checkpoint")
    bank_path = _require_file(zeva.task_context.bank_path, "model.zeva.task_context.bank_path")
    readout_path = _require_file(zeva.task_context.readout_cache_path, "model.zeva.task_context.readout_cache_path")
    head_path = _require_file(zeva.task_context.retrieval_checkpoint, "model.zeva.task_context.retrieval_checkpoint")
    static_path = _require_file(zeva.task_context.static_episode_cache_path, "model.zeva.task_context.static_episode_cache_path")
    stats_path = _require_file(cfg.data.train.pretrained_norm_stats, "data.train.pretrained_norm_stats")
    semantic_path = _require_file(cfg.data.train.semantic_task_map_path, "data.train.semantic_task_map_path")

    phase_root = Path(str(zeva.cache.path)).expanduser().resolve()
    pim_root = Path(str(zeva.prepared_stage2.pim_cache_path)).expanduser().resolve()
    if not phase_root.is_dir() or not pim_root.is_dir():
        raise FileNotFoundError("phase cache and prepared PIM cache must both exist")
    phase_manifest = phase_root / "manifest.json"
    phase_index = phase_root / "episode_index.json"
    if not phase_manifest.is_file() or not phase_index.is_file():
        raise FileNotFoundError(f"incomplete phase/effect cache: {phase_root}")

    base_sha = checkpoint_sha256(base_ckpt)
    cte_sha = checkpoint_sha256(cte_ckpt)
    stats_sha = sha256_file(stats_path)
    semantic_sha = sha256_file(semantic_path)
    top_k = int(zeva.memory.pim_top_k)
    bit_size = int(zeva.memory.bit_size)
    phase_dim = int(zeva.prompt.phase_dim)
    effect_dim = int(zeva.prompt.effect_dim)
    global_dim = int(zeva.prompt.global_dim)
    if top_k != int(zeva.prompt.persistent_length) or bit_size != int(zeva.prompt.brief_length):
        raise ValueError("PIM/BIT sizes do not match CausalPrompt lengths")

    cte_payload = torch.load(cte_ckpt, map_location="cpu", weights_only=False)
    cte_semantic = cte_payload.get("semantic_task_identity") or {}
    if str(cte_semantic.get("sha256")) != semantic_sha:
        raise ValueError("semantic map differs from CTE checkpoint")
    video_size = [int(v) for v in cfg.data.train.video_size]
    if list(cte_payload.get("cte_vae_input_size", ())) != video_size:
        raise ValueError("CTE VAE input size differs from Stage-2 video size")
    if tuple(cte_payload.get("camera_keys", ())) != ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        raise ValueError("CTE camera order mismatch")

    pim = PreparedPIMCache(
        pim_root,
        expected={
            "retrieval_mode": "phase",
            "history_semantics": "full_episode_prefix",
            "query_step_unit": "raw_action_step",
            "phase_dim": phase_dim,
            "effect_dim": effect_dim,
            "top_k": top_k,
            "bit_size": bit_size,
            "cte_checkpoint_sha256": cte_sha,
            "dataset_stats_sha256": stats_sha,
            "dataset_path": str(cfg.data.train.dataset_dirs[0]),
            "semantic_task_map_sha256": semantic_sha,
            "phase_cache_manifest_sha256": sha256_file(phase_manifest),
            "phase_cache_index_sha256": sha256_file(phase_index),
            "feature_dtype": "float32",
        },
    )
    if int(pim.manifest.get("semantic_task_count", -1)) != 50:
        raise ValueError("prepared PIM cache must contain 50 semantic tasks")
    validation = pim.manifest.get("validation") or {}
    if float(validation.get("samples", 0)) < 1:
        raise ValueError("prepared PIM cache has no recorded retrieval equivalence validation")

    static = StaticEpisodeContextCache(static_path)
    validate_static_artifact_hashes(static, bank_path=bank_path, head_path=head_path, readout_path=readout_path)
    for key, expected in {
        "base_checkpoint_sha256": base_sha,
        "cte_checkpoint_sha256": cte_sha,
        "dataset_stats_sha256": stats_sha,
        "semantic_task_map_sha256": semantic_sha,
        "video_size": video_size,
        "context_len": int(cfg.model.tokenizer_max_len),
        "top_k": int(zeva.task_context.top_k),
    }.items():
        actual = static.metadata.get(key)
        if actual != expected:
            raise ValueError(f"static context identity mismatch for {key}: {actual!r} != {expected!r}")
    if static.value_dim != global_dim:
        raise ValueError(f"static context dim={static.value_dim} != prompt.global_dim={global_dim}")
    if float(static.metadata.get("validation_samples", 0)) < 1:
        raise ValueError("static context cache has no recorded online-retriever equivalence validation")

    base = instantiate(cfg.data.train)
    identity = getattr(base, "semantic_task_identity", None)
    if identity is None or str(identity.get("sha256")) != semantic_sha:
        raise ValueError("RobotVideoDataset semantic identity mismatch")
    dataset = PreparedZevaStage2Dataset(base, pim, static)
    if len(dataset) != int(pim.manifest["num_queries"]):
        raise RuntimeError("prepared dataset/query count mismatch")

    count = _int_env("ZEVA_STAGE2_VALIDATE_SAMPLES", 64)
    indices = _sample_indices(len(dataset), count, int(cfg.get("seed", 42)))
    expected_h, expected_w = video_size
    context_len = int(cfg.model.tokenizer_max_len)
    text_dim = int(cfg.model.video_dit_config.text_dim)
    for ordinal, index in enumerate(indices, start=1):
        sample = dataset[index]
        _assert_tensor("video", sample["video"], (3, 9, expected_h, expected_w))
        _assert_tensor("action", sample["action"], (32, 14))
        _assert_tensor("context", sample["context"], (context_len, text_dim))
        _assert_tensor("context_mask", sample["context_mask"], (context_len,), dtype=torch.bool)
        _assert_tensor("phase", sample["phase"], (phase_dim,), dtype=torch.float32)
        _assert_tensor("bit_effects", sample["bit_effects"], (bit_size, effect_dim), dtype=torch.float32)
        _assert_tensor("bit_mask", sample["bit_mask"], (bit_size,), dtype=torch.bool)
        _assert_tensor("pim_phases", sample["pim_phases"], (top_k, phase_dim), dtype=torch.float32)
        _assert_tensor("pim_effects", sample["pim_effects"], (top_k, effect_dim), dtype=torch.float32)
        _assert_tensor("pim_mask", sample["pim_mask"], (top_k,), dtype=torch.bool)
        _assert_tensor("task_context", sample["task_context"], (global_dim,), dtype=torch.float32)
        if not bool(sample["pim_mask"].all()):
            raise ValueError(f"formal same-task PIM row {index} is unexpectedly padded")
        if ordinal % 16 == 0 or ordinal == len(indices):
            print(f"sample contract: {ordinal}/{len(indices)}")

    print("========== prepared Stage-2 validation ==========")
    print(f"queries: {len(dataset):,}")
    print(f"episodes: {len(pim.episode_ids):,}")
    print(f"semantic tasks: {len(pim.task_ids)}")
    print(f"sample contracts checked: {len(indices)}")
    print("artifact identities / shapes / dtypes / finiteness / dataset alignment: PASSED")

    if _bool_env("ZEVA_STAGE2_MODEL_SMOKE", True):
        if not torch.cuda.is_available():
            raise RuntimeError("model smoke requested but CUDA is unavailable")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        model = instantiate(
            cfg.model,
            model_dtype=_mixed_precision_to_model_dtype(str(cfg.mixed_precision)),
            device=str(device),
        )
        model.load_checkpoint(str(base_ckpt))
        model.zeva_task_context_identity = task_context_identity(
            "static", str(bank_path), str(head_path), int(zeva.task_context.top_k)
        )
        # DataLoader supplies the exact batch shapes used by Wan22Trainer.
        smoke = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
        smoke = dict(smoke)
        smoke["_training_mode"] = "zeva_stage2"
        model.train()
        amp_dtype = _mixed_precision_to_model_dtype(str(cfg.mixed_precision))
        autocast_enabled = amp_dtype in {torch.float16, torch.bfloat16}
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=autocast_enabled):
            loss, metrics = model(smoke)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Stage-2 model smoke produced non-finite loss: {loss}")
        for name, value in metrics.items():
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue
            if not torch.isfinite(torch.tensor(scalar)):
                raise FloatingPointError(f"Stage-2 smoke metric {name} is non-finite: {scalar}")
        if _bool_env("ZEVA_STAGE2_BACKWARD_SMOKE", True):
            trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
            if not trainable:
                raise RuntimeError("Stage-2 model smoke found no trainable addon parameters")
            loss.backward()
            finite_grad_count = 0
            nonzero_grad_count = 0
            for name, parameter in trainable:
                grad = parameter.grad
                if grad is None:
                    continue
                if not bool(torch.isfinite(grad).all()):
                    raise FloatingPointError(f"non-finite Stage-2 smoke gradient: {name}")
                finite_grad_count += 1
                if bool(grad.detach().abs().max() > 0):
                    nonzero_grad_count += 1
            if finite_grad_count == 0 or nonzero_grad_count == 0:
                raise RuntimeError(
                    f"Stage-2 backward smoke produced no usable addon gradients: "
                    f"finite={finite_grad_count}, nonzero={nonzero_grad_count}"
                )
            print(
                f"addon backward smoke: trainable={len(trainable)} "
                f"finite_grad={finite_grad_count} nonzero_grad={nonzero_grad_count}"
            )
        print(f"model forward smoke loss: {float(loss.detach()):.6f}")
        print("FROZEN FASTWAM + ZEVA STAGE-2 FORWARD/BACKWARD: PASSED")

    report = {
        "base_checkpoint_sha256": base_sha,
        "cte_checkpoint_sha256": cte_sha,
        "semantic_task_map_sha256": semantic_sha,
        "dataset_stats_sha256": stats_sha,
        "prepared_pim_manifest_sha256": sha256_file(pim_root / "manifest.json"),
        "static_context_sha256": checkpoint_sha256(static_path),
        "queries": len(dataset),
        "episodes": len(pim.episode_ids),
        "semantic_tasks": len(pim.task_ids),
        "sample_contracts_checked": len(indices),
        "model_smoke": _bool_env("ZEVA_STAGE2_MODEL_SMOKE", True),
        "backward_smoke": _bool_env("ZEVA_STAGE2_BACKWARD_SMOKE", True),
    }
    report_path = pim_root / "stage2_preflight_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"report: {report_path}")
    print("PREPARED STAGE-2 VALIDATION: PASSED")


if __name__ == "__main__":
    main()
