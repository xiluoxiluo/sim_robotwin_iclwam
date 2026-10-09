"""Validate cached Stage-1 Wan-VAE windows against the online RGB->VAE path."""

from __future__ import annotations

import random
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from fastwam.datasets.zeva_robotwin_dataset import ZevaRobotWinDataset
from fastwam.zeva.cte_latent_cache import CachedCTELatentWindowDataset
from fastwam.zeva.schemas import sha256_file
from fastwam.zeva.vae_adapter import FastWAMCTELatentEncoder, load_frozen_wan_vae


def _cfg_dict(value) -> dict:
    return {} if value is None else dict(OmegaConf.to_container(value, resolve=True))


def _video_size_hw(cfg: DictConfig) -> tuple[int, int]:
    return tuple(int(v) for v in cfg.data.train.video_size)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("validation requires one CUDA GPU")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)

    zeva = cfg.model.get("zeva", {})
    cte_values = _cfg_dict(zeva.get("cte"))
    cache_cfg = _cfg_dict(cfg.get("cte_latent_cache"))
    cache_value = cache_cfg.get("path", cte_values.get("latent_cache_path"))
    if cache_value in (None, "", "None", "null"):
        raise ValueError("set model.zeva.cte.latent_cache_path or cte_latent_cache.path")

    stats_path = Path(str(cfg.data.train.pretrained_norm_stats))
    semantic_path = Path(str(cfg.data.train.semantic_task_map_path))
    video_size = _video_size_hw(cfg)

    cache = CachedCTELatentWindowDataset(
        str(cache_value),
        expected_dataset_stats_sha256=sha256_file(stats_path),
        expected_semantic_task_sha256=sha256_file(semantic_path),
        expected_video_size=video_size,
        expected_action_dim=int(cte_values.get("action_dim", 14)),
        expected_transition_steps=int(cte_values.get("transition_steps", 4)),
        expected_latent_channels=int(cte_values.get("image_channels", 48)),
    )

    # Match the generation input path exactly.
    OmegaConf.update(cfg, "data.train.video_backend", "pyav", force_add=True)
    OmegaConf.update(cfg, "data.train.use_text_embed_cache", False, force_add=True)
    BaseLerobotDataset.presample_images = bool(cache.manifest.get("presample_images", True))

    base = instantiate(cfg.data.train)
    online_dataset = ZevaRobotWinDataset(base)

    model_values = _cfg_dict(cfg.model)
    vae, actual_vae_metadata = load_frozen_wan_vae(
        model_id=str(model_values.get("model_id", "Wan-AI/Wan2.2-TI2V-5B")),
        tokenizer_model_id=str(
            model_values.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B")
        ),
        device=str(device),
        torch_dtype=torch.bfloat16,
        redirect_common_files=bool(model_values.get("redirect_common_files", True)),
    )
    if dict(actual_vae_metadata) != dict(cache.vae_metadata):
        raise ValueError(
            f"VAE metadata mismatch:\ncache={cache.vae_metadata}\nactual={actual_vae_metadata}"
        )

    encoder = FastWAMCTELatentEncoder(
        vae,
        resize=video_size,
        expected_channels=int(cte_values.get("image_channels", 48)),
        input_range="minus_one_one",
    ).encode_history

    sample_count = min(int(cache_cfg.get("validate_samples", 32)), len(cache.rows))
    rng = random.Random(int(cfg.get("seed", 42)))
    selected = rng.sample(cache.rows, sample_count)

    worst_max_abs = 0.0
    worst_mean_abs = 0.0
    with torch.inference_mode():
        for row in tqdm(selected, desc="Validating latent cache", dynamic_ncols=True):
            dataset_index = int(row["dataset_index"])
            cached = cache[dataset_index]
            online = online_dataset[dataset_index]

            frames = online.get("cte_frames")
            if frames is None:
                frames = torch.cat(
                    (online["before_frames"], online["after_frames"][-1:]), dim=0
                )
            online_latent = encoder(frames.unsqueeze(0).to(device))[0].cpu()
            online_bf16 = online_latent.to(torch.bfloat16)
            cached_bf16 = cached["cte_frames"]

            if not torch.equal(online_bf16, cached_bf16):
                diff = (online_bf16.float() - cached_bf16.float()).abs()
                raise AssertionError(
                    "latent cache is not bit-exact after BF16 storage for "
                    f"dataset_index={dataset_index}: "
                    f"max_abs={float(diff.max())}, mean_abs={float(diff.mean())}"
                )

            diff = (online_latent.float() - cached_bf16.float()).abs()
            worst_max_abs = max(worst_max_abs, float(diff.max()))
            worst_mean_abs = max(worst_mean_abs, float(diff.mean()))

            if not torch.equal(
                online["transition_actions"].float().cpu(),
                cached["transition_actions"],
            ):
                raise AssertionError(f"action mismatch for dataset_index={dataset_index}")
            if not torch.equal(online["frame_valid"].cpu(), cached["frame_valid"]):
                raise AssertionError(f"frame_valid mismatch for dataset_index={dataset_index}")
            if not torch.equal(
                online["transition_valid"].cpu(), cached["transition_valid"]
            ):
                raise AssertionError(
                    f"transition_valid mismatch for dataset_index={dataset_index}"
                )
            if str(online["episode"].task_id) != str(cached["episode"].task_id):
                raise AssertionError(f"semantic task mismatch for dataset_index={dataset_index}")

    print("========== Latent cache validation ==========")
    print(f"samples checked: {sample_count}")
    print("BF16 bit equality: PASSED")
    print("actions/masks/semantic identity: PASSED")
    print(f"float32-online vs BF16-cache worst max abs: {worst_max_abs:.8f}")
    print(f"float32-online vs BF16-cache worst mean abs: {worst_mean_abs:.8f}")
    print("LATENT CACHE VALIDATION: PASSED")


if __name__ == "__main__":
    main()
