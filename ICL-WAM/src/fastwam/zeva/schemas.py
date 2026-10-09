"""Typed transition/cache schemas and exact RoboTwin temporal alignment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True)
class TransitionRecord:
    episode_id: str
    task_id: int | str
    task_name: str
    instruction: str
    episode_step: int
    episode_num_steps: int
    attempt_id: int = 0


@dataclass(frozen=True)
class CacheManifest:
    schema_version: str = "zeva_fastwam_robotwin_cache_v3"
    # v4 caches are computed from a contiguous episode prefix. v3 remains
    # readable for diagnostics/tests but must not be used for the repaired
    # Stage 2 training path.
    history_semantics: str = "window_local"
    query_step_unit: str = "episode_step"
    cte_checkpoint_sha256: str = ""
    dataset_stats_sha256: str = ""
    dataset_path: str = ""
    camera_keys: tuple[str, ...] = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    action_dim: int = 14
    action_group_size: int = 4
    action_horizon: int = 32
    video_frames: int = 9
    phase_dim: int = 128
    effect_dim: int = 128
    image_channels: int = 3
    # ``latent_channels`` is zero for direct RGB input and equals
    # ``image_channels`` for the released Zeva Wan-VAE serving path.
    latent_channels: int = 0
    cte_input_type: str = "rgb_frame"
    vae_metadata: dict[str, object] = field(default_factory=dict)
    cte_vae_input_size: tuple[int, int] | None = None
    effect_window_transitions: int = 4
    transition_count: int = 8
    feature_dtype: str = "float32"
    action_video_freq_ratio: int = 4
    action_normalization: str = "fastwam_processor_output"

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera_keys", tuple(self.camera_keys))
        object.__setattr__(self, "vae_metadata", dict(self.vae_metadata))
        if self.cte_vae_input_size is not None:
            size = tuple(int(value) for value in self.cte_vae_input_size)
            if len(size) != 2 or min(size) < 1:
                raise ValueError("cte_vae_input_size must be [H, W] with positive dimensions")
            object.__setattr__(self, "cte_vae_input_size", size)
        if self.schema_version not in {
            "zeva_fastwam_robotwin_cache_v2",
            "zeva_fastwam_robotwin_cache_v3",
            "zeva_fastwam_robotwin_cache_v4",
        }:
            raise ValueError(f"unsupported cache schema_version: {self.schema_version}")
        if self.history_semantics not in {"window_local", "full_episode_prefix"}:
            raise ValueError("unsupported cache history_semantics")
        if self.query_step_unit not in {"episode_step", "raw_action_step"}:
            raise ValueError("unsupported cache query_step_unit")
        if self.schema_version == "zeva_fastwam_robotwin_cache_v4" and (
            self.history_semantics != "full_episode_prefix"
            or self.query_step_unit != "raw_action_step"
        ):
            raise ValueError("cache v4 requires full_episode_prefix/raw_action_step semantics")
        if self.action_dim < 1 or self.action_group_size < 1 or self.action_horizon < 1:
            raise ValueError("cache action dimensions must be positive")
        if self.action_horizon % self.action_group_size != 0:
            raise ValueError("action_horizon must be divisible by action_group_size")
        if self.video_frames != self.action_horizon // self.action_group_size + 1:
            raise ValueError("video_frames is inconsistent with action_horizon/action_group_size")
        if self.phase_dim < 1 or self.effect_dim < 1 or self.image_channels < 1:
            raise ValueError("cache feature dimensions must be positive")
        if self.latent_channels < 0 or self.effect_window_transitions < 1:
            raise ValueError("latent_channels/effect_window_transitions are invalid")
        if self.cte_input_type not in {"rgb_frame", "wan_vae_latent"}:
            raise ValueError("cte_input_type must be 'rgb_frame' or 'wan_vae_latent'")
        if self.cte_input_type == "rgb_frame" and (self.latent_channels != 0 or self.image_channels != 3):
            raise ValueError("rgb_frame cache requires image_channels=3 and latent_channels=0")
        if self.cte_input_type == "wan_vae_latent" and self.latent_channels != self.image_channels:
            raise ValueError("wan_vae_latent cache requires latent_channels=image_channels")
        if self.cte_input_type == "wan_vae_latent" and not self.vae_metadata:
            raise ValueError("wan_vae_latent cache must record VAE metadata")
        if self.cte_input_type == "wan_vae_latent":
            required_vae_metadata = {
                "model_id",
                "vae_path",
                "z_dim",
                "temporal_downsample_factor",
                "upsampling_factor",
            }
            if not required_vae_metadata.issubset(self.vae_metadata):
                raise ValueError(
                    "wan_vae_latent cache must record complete VAE identity "
                    f"metadata: {sorted(required_vae_metadata)}"
                )
        if self.transition_count != self.action_horizon // self.action_group_size:
            raise ValueError("transition_count is inconsistent with action_horizon/action_group_size")
        if self.action_video_freq_ratio < 1:
            raise ValueError("action_video_freq_ratio must be positive")
        if self.feature_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("unsupported cache feature_dtype")
        if self.action_normalization != "fastwam_processor_output":
            raise ValueError("V1 requires FastWAM processor action normalization")
        if self.camera_keys != ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            raise ValueError("V1 requires camera order cam_high, cam_left_wrist, cam_right_wrist")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def validate(self, expected: dict[str, object] | None = None) -> None:
        values = self.to_dict()
        for key, value in (expected or {}).items():
            if values.get(key) != value:
                raise ValueError(f"cache manifest mismatch for {key}: expected {value!r}, got {values.get(key)!r}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transition_valid_mask(frame_valid: Tensor, action_valid: Tensor, action_group_size: int = 4) -> Tensor:
    if frame_valid.ndim != 2 or action_valid.ndim != 2 or action_valid.shape[1] % action_group_size:
        raise ValueError("frame_valid/action_valid shapes are invalid")
    transitions = action_valid.shape[1] // action_group_size
    if frame_valid.shape[1] != transitions + 1:
        raise ValueError("frame count must equal transition count + 1")
    grouped_actions = action_valid.reshape(action_valid.shape[0], transitions, action_group_size)
    # Preserve action-level validity.  CTE reduces this with ``all(dim=-1)``
    # only when a complete four-control transition is needed.
    frame_pair = (frame_valid[:, :-1] & frame_valid[:, 1:]).unsqueeze(-1)
    return grouped_actions & frame_pair


def build_transition_view(actions: Tensor, video_frames: Tensor, *, frame_valid: Tensor | None = None, action_valid: Tensor | None = None, action_group_size: int = 4) -> dict[str, Tensor]:
    if actions.ndim != 3 or video_frames.ndim < 2 or actions.shape[1] % action_group_size:
        raise ValueError("expected actions [B,32,A] and video frames [B,9,...]")
    transitions = actions.shape[1] // action_group_size
    if video_frames.shape[1] != transitions + 1:
        raise ValueError("video frames must align to actions/action_group_size")
    frame_valid = torch.ones((actions.shape[0], video_frames.shape[1]), dtype=torch.bool, device=actions.device) if frame_valid is None else frame_valid.bool()
    action_valid = torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device) if action_valid is None else action_valid.bool()
    transition_valid = transition_valid_mask(frame_valid, action_valid, action_group_size)
    return {
        "transition_actions": actions.reshape(actions.shape[0], transitions, action_group_size, actions.shape[-1]),
        "before_frames": video_frames[:, :-1],
        "after_frames": video_frames[:, 1:],
        "transition_valid": transition_valid,
    }


def save_manifest(path: str | Path, manifest: CacheManifest) -> None:
    Path(path).write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")


def load_manifest(path: str | Path, expected: dict[str, object] | None = None) -> CacheManifest:
    manifest = CacheManifest(**json.loads(Path(path).read_text()))
    manifest.validate(expected)
    return manifest
