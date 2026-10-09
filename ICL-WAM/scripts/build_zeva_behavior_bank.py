"""Build CTE demonstration prototypes and frozen FastWAM initial readouts."""

from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fastwam.datasets.zeva_robotwin_dataset import ZevaRobotWinDataset
from fastwam.runtime import _mixed_precision_to_model_dtype
from fastwam.zeva import CausalTransitionEncoder, CausalTransitionEncoderConfig, FastWAMCTELatentEncoder, validate_vae_metadata
from fastwam.zeva.checkpoint import checkpoint_sha256, load_cte_checkpoint
from fastwam.zeva.static_task_context import READOUT_FORMAT, READOUT_KIND
from fastwam.zeva.task_context_data import build_behavior_bank


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    if not torch.cuda.is_available():
        raise RuntimeError("Real FastWAM readout extraction requires CUDA; CPU tests use small fixtures")
    if not bool(cfg.data.train.get("is_training_set", False)):
        raise ValueError("Build task behavior prototypes from data.train only")
    if int(cfg.data.train.global_sample_stride) != 1 or int(cfg.data.train.action_video_freq_ratio) != 4:
        raise ValueError("behavior bank requires raw stride=1 and four actions per visual transition")
    tc = cfg.model.zeva.task_context
    if int(cfg.data.train.context_len) != int(cfg.model.tokenizer_max_len):
        raise ValueError("training text context_len must match deployment tokenizer_max_len")
    for name, value in {
        "ckpt": cfg.get("ckpt"), "CTE checkpoint": cfg.model.zeva.cte.checkpoint,
        "normalization stats": cfg.data.train.pretrained_norm_stats,
    }.items():
        if value is None or not Path(str(value)).is_file():
            raise FileNotFoundError(f"Missing {name}: {value}")
    for name in ("bank_path", "readout_cache_path"):
        if tc.get(name) in (None, "", "None", "null"):
            raise ValueError(f"Set model.zeva.task_context.{name}")
    if Path(str(tc.bank_path)).resolve() == Path(str(tc.readout_cache_path)).resolve():
        raise ValueError("bank and readout cache must use different output files")
    cte_path = str(cfg.model.zeva.cte.checkpoint)
    payload = torch.load(cte_path, map_location="cpu", weights_only=False)
    cte_config = payload.get("model_config", payload.get("config", {}))
    cte_config = cte_config.get("cte", cte_config)
    allowed = CausalTransitionEncoderConfig.__dataclass_fields__
    cte = CausalTransitionEncoder(CausalTransitionEncoderConfig(**{k: v for k, v in cte_config.items() if k in allowed}))
    load_cte_checkpoint(cte_path, cte)
    if cte.cfg.action_dim != 14 or cte.cfg.transition_steps != 4:
        raise ValueError("CTE checkpoint must use RoboTwin action_dim=14, transition_steps=4")
    if tuple(payload.get("camera_keys", ())) != ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        raise ValueError("CTE checkpoint camera order is incompatible with RoboTwin")
    if cte.cfg.retrieval_dim != int(tc.key_dim) or cte.cfg.hidden_dim != int(tc.value_dim):
        raise ValueError("bank key_dim/value_dim must match CTE retrieval_dim/hidden_dim")
    video_size = [int(value) for value in cfg.data.train.video_size]
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.zeva.enabled = False
    policy = instantiate(model_cfg, model_dtype=_mixed_precision_to_model_dtype(str(cfg.mixed_precision)), device="cuda")
    policy.load_checkpoint(str(cfg.ckpt))
    policy.eval().requires_grad_(False)
    cte.to("cuda").eval().requires_grad_(False)
    input_type = payload.get("cte_input_type", "rgb_frame")
    frame_encoder = None
    if input_type == "wan_vae_latent":
        if list(payload.get("cte_vae_input_size", ())) != video_size:
            raise ValueError("CTE checkpoint and bank video size differ")
        validate_vae_metadata(payload.get("vae_metadata", {}), {
            "model_id": str(cfg.model.model_id),
            "z_dim": int(getattr(policy.vae, "z_dim", -1)),
            "temporal_downsample_factor": int(getattr(policy.vae, "temporal_downsample_factor", -1)),
            "upsampling_factor": int(getattr(policy.vae, "upsampling_factor", -1)),
        })
        frame_encoder = FastWAMCTELatentEncoder(
            policy, resize=tuple(video_size), expected_channels=cte.cfg.image_channels,
            input_range="minus_one_one",
        ).encode_history
    elif input_type != "rgb_frame" or cte.cfg.image_channels != 3:
        raise ValueError("unsupported CTE input representation")
    bank, readouts = build_behavior_bank(
        ZevaRobotWinDataset(instantiate(cfg.data.train)), cte, policy,
        frame_encoder=frame_encoder, temperature=float(tc.temperature),
        metadata={
            "base_checkpoint_sha256": checkpoint_sha256(str(cfg.ckpt)),
            "cte_checkpoint_sha256": checkpoint_sha256(cte_path),
            "dataset_stats_sha256": checkpoint_sha256(str(cfg.data.train.pretrained_norm_stats)),
            "video_size": video_size, "context_len": int(cfg.model.tokenizer_max_len),
            "dataset_dirs": list(cfg.data.train.dataset_dirs),
        },
    )
    for path in (tc.bank_path, tc.readout_cache_path):
        Path(str(path)).parent.mkdir(parents=True, exist_ok=True)
    bank.save(str(tc.bank_path))
    torch.save({
        "format": READOUT_FORMAT, "readout_kind": READOUT_KIND,
        "bank_sha256": checkpoint_sha256(str(tc.bank_path)),
        "episode_ids": [entry["episode_id"] for entry in bank.entries],
        "readouts": readouts,
    }, str(tc.readout_cache_path))
    print(f"Wrote {len(bank)} CTE behavior prototypes and initial readouts {tuple(readouts.shape)}")


if __name__ == "__main__":
    main()
