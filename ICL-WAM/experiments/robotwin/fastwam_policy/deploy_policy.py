import logging
import os
import sys
import time
import inspect
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.zeva import (
    CausalCTEHistory,
    CausalMemoryLifecycle,
    FastWAMCTELatentEncoder,
    LifecycleConfig,
    CausalTransitionEncoder,
    CausalTransitionEncoderConfig,
    validate_vae_metadata,
    PersistentInteractionMemory,
    PersistentInteractionMemoryConfig,
    task_tokens_from_context,
    TaskContextBank,
    retrieve_task_context,
)
from fastwam.zeva.checkpoint import checkpoint_sha256, load_cte_checkpoint
from fastwam.zeva.static_task_context import (
    StaticTaskContextRetriever, StaticTaskContextSession, task_context_identity,
)

logger = logging.getLogger(__name__)


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(
                f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}"
            ) from exc
        return relative.as_posix()

    if _is_none_like(sim_cfg_name):
        return "sim_robotwin.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
) -> DictConfig:
    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. "
            "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    # Match RobotVideoDataset exactly: tensor bilinear resize with antialias
    # enabled (the processor first maps each raw camera to 240x320, then the
    # RoboTwin mosaic maps it to its 256/128-pixel tile).
    import torchvision.transforms.functional as transforms_F

    array = np.asarray(image)
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    if tensor.dtype == torch.uint8:
        tensor = tensor.float().div(255.0)
    else:
        tensor = tensor.float()
        if float(tensor.detach().amax()) > 1.0:
            tensor = tensor.div(255.0)
    resized = transforms_F.resize(
        tensor,
        size=[int(size_wh[1]), int(size_wh[0])],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    return resized.permute(1, 2, 0).contiguous().numpy()


class WorldActionRobotWinPolicy:
    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        timing_enabled: bool,
        num_video_frames: int,
        video_size: tuple[int, int] | list[int] = (384, 320),
        zeva_mode: str = "base",
        cte_checkpoint: Optional[str] = None,
        addon_checkpoint: Optional[str] = None,
        pim_top_k: int = 4,
        max_attempts: int = 1,
    ) -> None:
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        model_cfg_copy.load_text_encoder = True

        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device).eval()
        requested_zeva_mode = str(zeva_mode)
        if requested_zeva_mode not in {
            "base",
            "zeva_stage2",
            "pim_shadow",
            "pim_on",
        }:
            raise ValueError(
                "zeva_mode must be base, zeva_stage2, pim_shadow, or pim_on"
            )
        zeva_memory_cfg = model_cfg_copy.get("zeva", {}).get("memory", {})
        task_context_cfg = model_cfg_copy.get("zeva", {}).get("task_context", {})
        self.task_context_bank = None
        self._static_task_context_session = None
        self._last_task_context_retrieval = None
        self.task_context_mode = str(task_context_cfg.get("mode", "static"))
        self.task_context_top_k = int(task_context_cfg.get("top_k", 5))
        if requested_zeva_mode not in {"base", "pim_shadow"} and self.task_context_mode == "bank":
            if self.model.zeva_prompt_encoder is None:
                raise ValueError("task-context bank requires an enabled Zeva prompt encoder")
            bank_path = task_context_cfg.get("bank_path")
            if _is_none_like(bank_path):
                raise ValueError("zeva.task_context.bank_path is required when mode=bank")
            self.task_context_bank = TaskContextBank.load(
                str(bank_path),
                expected_key_dim=int(task_context_cfg.get("key_dim", 256)),
                expected_value_dim=int(
                    task_context_cfg.get(
                        "value_dim",
                        self.model.zeva_prompt_encoder.config.global_dim,
                    )
                ),
            )
            if self.task_context_bank.value_dim != int(self.model.zeva_prompt_encoder.config.global_dim):
                raise ValueError("zeva.task_context.value_dim must match zeva.prompt.global_dim")
            if self.task_context_top_k < 1 or self.task_context_top_k > len(self.task_context_bank):
                raise ValueError(
                    "zeva.task_context.top_k must be within the task-context bank size; "
                    f"got {self.task_context_top_k}, bank_size={len(self.task_context_bank)}"
                )
        elif requested_zeva_mode not in {"base", "pim_shadow"} and self.task_context_mode == "static":
            for name in ("bank_path", "retrieval_checkpoint"):
                if _is_none_like(task_context_cfg.get(name)):
                    raise ValueError(f"static task context requires zeva.task_context.{name}")
            if _is_none_like(cte_checkpoint):
                raise ValueError("static task context requires the matching CTE checkpoint")
            bank_path = str(task_context_cfg["bank_path"])
            head_path = str(task_context_cfg["retrieval_checkpoint"])
            retriever = StaticTaskContextRetriever.load(
                bank_path, head_path, top_k=self.task_context_top_k, device=device,
                expected={
                    "base_checkpoint_sha256": checkpoint_sha256(checkpoint_path),
                    "cte_checkpoint_sha256": checkpoint_sha256(str(cte_checkpoint)),
                    "dataset_stats_sha256": checkpoint_sha256(dataset_stats_path),
                    "video_size": [int(value) for value in video_size],
                    "context_len": int(model_cfg_copy.get("tokenizer_max_len", 128)),
                    "readout_dim": int(model_cfg_copy.video_dit_config.hidden_dim),
                },
            )
            if self.model.zeva_prompt_encoder is None or retriever.bank.value_dim != self.model.zeva_prompt_encoder.config.global_dim:
                raise ValueError("static behavior bank value_dim must match the Zeva prompt encoder")
            self._static_task_context_session = StaticTaskContextSession(retriever)
            self.model.zeva_task_context_identity = task_context_identity(
                "static", bank_path, head_path, self.task_context_top_k,
            )
        elif requested_zeva_mode not in {"base", "pim_shadow"} and self.task_context_mode != "pooling":
            raise ValueError("zeva.task_context.mode must be pooling, bank, or static")

        self.zeva_mode = requested_zeva_mode
        self.cte = None
        self.lifecycle = None
        self._cte_history = None
        self._cte_frame_encoder = None
        self._observed_effect_count = 0
        self._transition_actions: list[torch.Tensor] = []
        self._attempt_finalized = False
        if len(video_size) != 2 or min(int(v) for v in video_size) < 1:
            raise ValueError(f"video_size must be [H, W] with positive dimensions, got {video_size}")
        self.video_size = tuple(int(v) for v in video_size)
        self.max_attempts = int(max_attempts)
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._needs_episode_reset = True
        if self.zeva_mode != "base":
            if not getattr(self.model, "zeva_enabled", False):
                raise ValueError("Zeva evaluation requires model config zeva.enabled=true")
            if not cte_checkpoint:
                raise ValueError("cte_checkpoint is required for Zeva evaluation")
            if requested_zeva_mode in {"zeva_stage2", "pim_on"} and not addon_checkpoint:
                raise ValueError(
                    "addon_checkpoint is required for zeva_stage2 and pim_on evaluation"
                )
            cte_payload = torch.load(str(cte_checkpoint), map_location="cpu", weights_only=False)
            cte_values = dict(cte_payload.get("config", cte_payload.get("model_config", {})))
            cte_values = dict(cte_values.get("cte", cte_values))
            checkpoint_action_dim = int(cte_payload.get("action_dim", cte_values.get("action_dim", -1)))
            if checkpoint_action_dim != 14 or tuple(cte_payload.get("camera_keys", ())) != ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                raise ValueError(
                    "CTE checkpoint metadata is incompatible with RoboTwin V1 "
                    "(action_dim=14, cameras=cam_high/cam_left_wrist/cam_right_wrist)"
                )
            allowed_cte = set(CausalTransitionEncoderConfig.__dataclass_fields__)
            cte_cfg = CausalTransitionEncoderConfig(**{k: v for k, v in cte_values.items() if k in allowed_cte})
            if (
                cte_cfg.action_dim != 14
                or cte_cfg.transition_steps != 4
                or cte_cfg.effect_window_transitions != 4
            ):
                raise ValueError(
                    "RoboTwin Zeva V1 requires CTE action_dim=14, "
                    "transition_steps=4, and effect_window_transitions=4; "
                    f"got action_dim={cte_cfg.action_dim}, "
                    f"transition_steps={cte_cfg.transition_steps}, "
                    f"effect_window_transitions={cte_cfg.effect_window_transitions}"
                )
            if cte_cfg.phase_dim != int(self.model.zeva_prompt_encoder.config.phase_dim) or cte_cfg.effect_dim != int(self.model.zeva_prompt_encoder.config.effect_dim):
                raise ValueError(
                    "CTE phase/effect dimensions must match the attached CausalPromptEncoder: "
                    f"cte=({cte_cfg.phase_dim},{cte_cfg.effect_dim}), "
                    f"prompt=({self.model.zeva_prompt_encoder.config.phase_dim},"
                    f"{self.model.zeva_prompt_encoder.config.effect_dim})"
                )
            prompt_cfg = self.model.zeva_prompt_encoder.config
            if int(prompt_cfg.persistent_length) != int(pim_top_k):
                raise ValueError(
                    "pim_top_k must match the trained CausalPromptEncoder persistent_length; "
                    f"got pim_top_k={pim_top_k}, persistent_length={prompt_cfg.persistent_length}"
                )
            configured_top_k = int(zeva_memory_cfg.get("pim_top_k", pim_top_k))
            if configured_top_k != int(pim_top_k):
                raise ValueError(
                    "Evaluation pim_top_k must match the trained model.zeva.memory.pim_top_k; "
                    f"got {pim_top_k} vs {configured_top_k}"
                )
            self.cte = CausalTransitionEncoder(cte_cfg).to(device).eval()
            load_cte_checkpoint(str(cte_checkpoint), self.cte, map_location=device)
            cte_input_type = str(cte_payload.get("cte_input_type", cte_values.get("input_type", "rgb_frame")))
            if cte_input_type == "rgb_frame":
                if cte_cfg.image_channels != 3:
                    raise ValueError("rgb_frame CTE checkpoints must set image_channels=3")
            elif cte_input_type == "wan_vae_latent":
                if int(cte_payload.get("latent_channels", cte_cfg.image_channels)) != cte_cfg.image_channels:
                    raise ValueError("CTE latent channel metadata does not match image_channels")
                vae_metadata = dict(cte_payload.get("vae_metadata", {}))
                required_vae_metadata = {
                    "model_id",
                    "vae_path",
                    "z_dim",
                    "temporal_downsample_factor",
                    "upsampling_factor",
                }
                if not required_vae_metadata.issubset(vae_metadata):
                    raise ValueError(
                        "wan_vae_latent CTE checkpoints must record complete "
                        f"VAE identity metadata: {sorted(required_vae_metadata)}"
                    )
                runtime_vae_metadata = {
                    "model_id": str(model_cfg.get("model_id", "")),
                    "z_dim": int(getattr(self.model.vae, "z_dim", -1)),
                    "temporal_downsample_factor": int(
                        getattr(self.model.vae, "temporal_downsample_factor", -1)
                    ),
                    "upsampling_factor": int(getattr(self.model.vae, "upsampling_factor", -1)),
                }
                if runtime_vae_metadata["model_id"]:
                    validate_vae_metadata(vae_metadata, runtime_vae_metadata)
                checkpoint_size = cte_payload.get("cte_vae_input_size")
                if checkpoint_size is None or len(checkpoint_size) != 2:
                    raise ValueError(
                        "wan_vae_latent CTE checkpoints must record cte_vae_input_size as [H, W]"
                    )
                checkpoint_size = tuple(int(v) for v in checkpoint_size)
                if checkpoint_size != self.video_size:
                    raise ValueError(
                        "CTE VAE input size mismatch: checkpoint declares "
                        f"{checkpoint_size}, evaluation data uses {self.video_size}"
                    )
                self._cte_frame_encoder = FastWAMCTELatentEncoder(
                    self.model,
                    resize=self.video_size,
                    expected_channels=cte_cfg.image_channels,
                    input_range="minus_one_one",
                ).encode_history
            else:
                raise ValueError(
                    f"Unsupported CTE input type {cte_input_type!r}; expected rgb_frame or wan_vae_latent"
                )
            if addon_checkpoint and requested_zeva_mode in {"zeva_stage2", "pim_on"}:
                self.model.load_zeva_addon_checkpoint(
                    str(addon_checkpoint),
                    base_checkpoint_sha256=checkpoint_sha256(checkpoint_path),
                    cte_checkpoint_sha256=checkpoint_sha256(cte_checkpoint),
                )
            pim = PersistentInteractionMemory(
                PersistentInteractionMemoryConfig(
                    phase_dim=cte_cfg.phase_dim,
                    effect_dim=cte_cfg.effect_dim,
                    capacity=int(zeva_memory_cfg.get("pim_max_entries", 64)),
                    top_k=int(pim_top_k),
                    merge_threshold=float(zeva_memory_cfg.get("merge_threshold", 0.85)),
                    phase_merge_weight=float(zeva_memory_cfg.get("beta_phase", 0.5)),
                    effect_merge_weight=float(zeva_memory_cfg.get("beta_effect", 0.5)),
                )
            )
            self.lifecycle = CausalMemoryLifecycle(
                pim,
                LifecycleConfig(
                    bit_size=int(prompt_cfg.brief_length),
                    transition_steps=cte_cfg.transition_steps,
                    effect_window_transitions=cte_cfg.effect_window_transitions,
                    action_dim=cte_cfg.action_dim,
                ),
            )

        self.processor: FastWAMProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.action_horizon = int(action_horizon)
        if self.zeva_mode != "base" and self.action_horizon != 32:
            raise ValueError(
                "Zeva V1 requires action_horizon=32 to preserve the trained 8x4 transition alignment"
            )
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        if self.zeva_mode != "base":
            transition_steps = int(self.cte.cfg.transition_steps)
            if self.replan_steps % transition_steps:
                raise ValueError(
                    "Zeva evaluation requires replan_steps to be a multiple of "
                    f"the CTE transition_steps ({transition_steps}); got {self.replan_steps}"
                )
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.timing_enabled = bool(timing_enabled)
        self._num_video_frames = int(num_video_frames)

        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = 0
        self.step_count = 0
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}

        logger.info(
            "Initialized WorldActionRobotWinPolicy | ckpt=%s | stats=%s | horizon=%d | replan=%d",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
        )

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action_key = action_meta[0]["key"]
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]
        action = action.to(dtype=torch.float32, device="cpu")
        # Reconstruct the processor's complete inverse (merger, normalizer,
        # and optional task-specific transforms), keeping action normalization
        # identical to the dataset and online CTE input path.
        dummy_state = torch.zeros(
            (action.shape[0], 1, state_meta[0]["shape"]), dtype=action.dtype
        )
        batch = {
            "action": {action_key: action},
            "state": {state_key: dummy_state},
        }
        batch = self.processor.action_state_merger.backward(batch)
        batch = self.processor.normalizer.backward(batch)
        if self.processor.action_state_transforms is not None:
            for transform in reversed(self.processor.action_state_transforms):
                batch = transform.backward(batch)
        return batch["action"][action_key].numpy()

    def _normalize_action(self, action: np.ndarray, state: Optional[np.ndarray] = None) -> torch.Tensor:
        action_meta = self.processor.shape_meta["action"]
        state_meta = self.processor.shape_meta["state"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        key = action_meta[0]["key"]
        state_key = state_meta[0]["key"]
        value = torch.as_tensor(action, dtype=torch.float32).reshape(1, -1)
        if value.shape[-1] != action_meta[0]["raw_shape"]:
            raise ValueError(f"Expected raw action dim {action_meta[0]['raw_shape']}, got {value.shape[-1]}")
        # Use the exact processor transform/normalizer path used by training.
        # The merger requires a state field, so provide a throwaway zero state;
        # action transforms are allowed to inspect it but no state is retained.
        state_value = torch.zeros((1, state_meta[0]["raw_shape"]), dtype=value.dtype) if state is None else torch.as_tensor(state, dtype=value.dtype).reshape(1, -1)
        if state_value.shape[-1] != state_meta[0]["raw_shape"]:
            raise ValueError(f"Expected raw state dim {state_meta[0]['raw_shape']}, got {state_value.shape[-1]}")
        batch = {"action": {key: value}, "state": {state_key: state_value}}
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        batch = self.processor.action_state_merger.forward(batch)
        return batch["action"][0]

    def _build_robotwin_image_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        obs_data = observation["observation"]
        required = ("head_camera", "left_camera", "right_camera")
        if any(key not in obs_data or "rgb" not in obs_data[key] for key in required):
            raise KeyError(
                "RoboTwin observation must expose RGB arrays under head_camera, "
                "left_camera, and right_camera"
            )
        for key in required:
            rgb = np.asarray(obs_data[key]["rgb"])
            if rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError(f"{key}.rgb must be an HxWx3 array, got {rgb.shape}")
        # The training path first applies the per-camera processor resize to
        # [240,320], then builds the fixed RoboTwin mosaic.  Repeating both
        # stages online prevents a silent train/deploy RGB distribution shift.
        head_raw = _resize_rgb(obs_data["head_camera"]["rgb"], (320, 240))
        left_raw = _resize_rgb(obs_data["left_camera"]["rgb"], (320, 240))
        right_raw = _resize_rgb(obs_data["right_camera"]["rgb"], (320, 240))
        head = _resize_rgb(head_raw, (320, 256))
        left = _resize_rgb(left_raw, (160, 128))
        right = _resize_rgb(right_raw, (160, 128))
        bottom = np.concatenate([left, right], axis=1)
        image = np.concatenate([head, bottom], axis=0)  # [384, 320, 3]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )
        image_tensor = image_tensor * 2.0 - 1.0
        return image_tensor

    def _infer_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        image_tensor = self._build_robotwin_image_tensor(observation)
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self._normalize_state(state_vector)

        prompt = DEFAULT_PROMPT.format(task=instruction)
        if self.zeva_mode != "base":
            if self._cte_history is None and self._needs_episode_reset:
                self.lifecycle.reset_episode(instruction, episode_id=f"episode-{self.episode_count}")
                self._needs_episode_reset = False
                self._cte_history = CausalCTEHistory(self.cte, frame_encoder=self._cte_frame_encoder)
                self._cte_history.reset(image_tensor[0].float())
                self._observed_effect_count = 0
            elif self._cte_history is None:
                self._cte_history = CausalCTEHistory(self.cte, frame_encoder=self._cte_frame_encoder)
                self._cte_history.reset(image_tensor[0].float())
                self._observed_effect_count = 0
            with torch.no_grad():
                context, _context_mask = self.model.encode_prompt(prompt)
                task_dim = int(self.model.zeva_prompt_encoder.config.global_dim)
                if self.zeva_mode == "pim_shadow":
                    # Shadow mode exercises CTE/BIT/PIM retrieval but must not
                    # depend on task-context artifacts or an addon checkpoint.
                    task_tokens = torch.zeros(
                        (1, task_dim), device=self.model.device, dtype=self.model.torch_dtype
                    )
                elif self._static_task_context_session is not None:
                    task_context_result = self._static_task_context_session.resolve(
                        self.model, image_tensor, context, _context_mask, instruction,
                    )
                    task_tokens = task_context_result.values
                    self._last_task_context_retrieval = {
                        "indices": task_context_result.indices.cpu(),
                        "scores": task_context_result.scores.cpu(),
                        "sources": task_context_result.sources,
                    }
                elif self.task_context_bank is None:
                    task_tokens = task_tokens_from_context(context, _context_mask, task_dim)
                else:
                    task_tokens, task_context_result = retrieve_task_context(
                        context,
                        _context_mask,
                        self.task_context_bank,
                        output_dim=task_dim,
                        top_k=self.task_context_top_k,
                    )
                    self._last_task_context_retrieval = {
                        "indices": task_context_result.indices.detach().cpu(),
                        "scores": task_context_result.scores.detach().cpu(),
                        "sources": task_context_result.sources,
                    }
                encoded = self._cte_history.forward()
                phase = encoded["phase"][:, -1]
                memory = self.lifecycle.memory_inputs(phase, task_tokens)
                if self.zeva_mode == "pim_shadow":
                    causal_prompt = torch.zeros(
                        (1, int(self.model.zeva_prompt_encoder.config.hidden_dim)),
                        device=self.model.device,
                        dtype=self.model.torch_dtype,
                    )
                    pim_mask = memory["pim_mask"]
                elif self.zeva_mode == "zeva_stage2":
                    causal_prompt = torch.zeros(
                        (1, int(self.model.zeva_prompt_encoder.config.hidden_dim)),
                        device=self.model.device,
                        dtype=self.model.torch_dtype,
                    )
                    pim_mask = torch.zeros_like(memory["pim_mask"], dtype=torch.bool)
                else:
                    causal_prompt = self.model.zeva_prompt_encoder(**memory)
                    pim_mask = memory["pim_mask"]
                zeva_action_residual = None
                if (
                    self.zeva_mode != "pim_shadow"
                    and getattr(self.model, "zeva_injection_mode", "memory_residual") == "exact_zeva"
                ):
                    adapter = self.model.zeva_behavior_prefix_adapter
                    prior_mean, _prior_std = adapter.prior(
                        task_tokens,
                        phase,
                        memory["bit_effects"],
                        memory["bit_mask"],
                    )
                    zeva_action_residual = adapter.action_prior_residual(
                        prior_mean, training=False
                    )
            infer_kwargs = {
                "prompt": None,
                "context": context,
                "context_mask": _context_mask,
                "causal_prompt": causal_prompt,
                "pim_mask": pim_mask,
                "zeva_action_residual": zeva_action_residual,
                "zeva_task_tokens": task_tokens,
                "zeva_mode": self.zeva_mode,
                "enable_prefix_injection": self.zeva_mode in {"zeva_stage2", "pim_on"},
                "enable_action_prior": self.zeva_mode in {"zeva_stage2", "pim_on"},
                "input_image": image_tensor,
                "action_horizon": self.action_horizon,
                "proprio": proprio,
                "negative_prompt": self.negative_prompt,
                "text_cfg_scale": self.text_cfg_scale,
                "num_inference_steps": self.num_inference_steps,
                "sigma_shift": self.sigma_shift,
                "seed": self.seed,
                "rand_device": self.rand_device,
                "tiled": self.tiled,
            }
            infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
            with torch.no_grad():
                pred = self.model.infer_action_zeva(**infer_kwargs)
            if self.timing_enabled:
                self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0
            return self._denormalize_action(pred["action"])[0]
        infer_kwargs = {
            "prompt": prompt,
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            infer_kwargs["num_video_frames"] = int(self._num_video_frames)
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        with torch.no_grad():
            pred = self.model.infer_action(**infer_kwargs)
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0

        action_tensor = pred["action"]  # [T, D]
        action_chunk = self._denormalize_action(action_tensor)[0]  # [T, D]
        return action_chunk

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        action_chunk = self._infer_action_chunk(observation=observation, instruction=instruction)
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        if self.zeva_mode != "base" and observation is not None and self._cte_history is not None:
            # The observation is the after-frame for the oldest complete group
            # of four actions.  RoboTwin supplies one observation per low-level
            # action; consume at most one group so an observation is never
            # incorrectly reused for a later group.
            next_image = self._build_robotwin_image_tensor(observation)[0].float()
            if len(self._transition_actions) >= self.cte.cfg.transition_steps:
                action_group = torch.stack(self._transition_actions[: self.cte.cfg.transition_steps]).to(self.model.device)
                self._cte_history.append_transition(action_group, next_image)
                self._transition_actions = self._transition_actions[self.cte.cfg.transition_steps :]
                encoded = self._cte_history.forward()
                complete = encoded["effect_complete"][0]
                while self._observed_effect_count < complete.shape[0]:
                    effect_index = self._observed_effect_count
                    if bool(complete[effect_index]):
                        start = effect_index * self.cte.cfg.effect_window_transitions
                        self.lifecycle.observe_completed_effect(
                            encoded["phase"][0, start],
                            encoded["effect_post"][0, effect_index],
                            metadata={"source_step": start * self.cte.cfg.transition_steps},
                        )
                    self._observed_effect_count += 1
        if not self.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty "
                    "(replan step for fastwam)."
                )
            instruction = task_env.get_instruction()
            self._fill_action_queue(observation=observation, instruction=instruction)

        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self.zeva_mode != "base":
            raw_state = None if observation is None else np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
            self._transition_actions.append(self._normalize_action(action, raw_state))
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0

    def get_timing_rollout(self) -> Dict[str, float]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
        }

    def reset(self) -> None:
        if getattr(self, "_static_task_context_session", None) is not None:
            self._static_task_context_session.reset()
        self._last_task_context_retrieval = None
        self.pending_actions.clear()
        self.episode_count += 1
        self.step_count = 0
        self.reset_timing_rollout()
        self._cte_history = None
        self._observed_effect_count = 0
        self._transition_actions.clear()
        self._attempt_finalized = False
        if self.zeva_mode != "base":
            # The evaluator calls reset for both new episodes and retries.  A
            # retry explicitly follows with begin_attempt(), which preserves
            # the current episode's PIM; otherwise the next inference starts a
            # fresh episode and clears PIM.
            self._needs_episode_reset = True

    def begin_attempt(self, attempt_id: int) -> None:
        """Reset transient CTE/BIT state while retaining episode-scoped PIM."""
        if getattr(self, "_static_task_context_session", None) is not None:
            self._static_task_context_session.reset()
        self._last_task_context_retrieval = None
        if self.lifecycle is None:
            return
        attempt_id = int(attempt_id)
        if attempt_id == 0:
            # Attempt zero always starts a new fixed episode.  The lifecycle is
            # initialized lazily once the first observation/instruction arrives.
            self._cte_history = None
            self._observed_effect_count = 0
            self._transition_actions.clear()
            self._attempt_finalized = False
            self._needs_episode_reset = True
            return
        if self.lifecycle.pim.task_cluster is not None:
            self.lifecycle.reset_attempt(attempt_id)
        self._cte_history = None
        self._observed_effect_count = 0
        self._transition_actions.clear()
        self._attempt_finalized = False
        self._needs_episode_reset = self.lifecycle.pim.task_cluster is None

    def finalize_attempt(self, task_env, success: bool = False) -> None:
        """Finalize one normally completed attempt, including failures.

        RoboTwin performs the terminal ``get_obs()`` inside ``take_action``
        before setting ``eval_success``; the evaluator exits its control loop
        immediately afterwards, so the normal next-step update would otherwise
        never see this final after-frame.
        """
        if self._attempt_finalized:
            return

        # Base mode has no lifecycle state, but still clear transient actions
        # so a policy object can be safely reused by the evaluator.
        if self.zeva_mode == "base" or self.cte is None or self.lifecycle is None:
            self._transition_actions.clear()
            self._attempt_finalized = True
            return

        try:
            terminal_obs = getattr(task_env, "now_obs", None)
            has_terminal_obs = isinstance(terminal_obs, dict) and "observation" in terminal_obs
            has_complete_tail = len(self._transition_actions) == self.cte.cfg.transition_steps

            # Only a complete four-action group paired with a trustworthy
            # after-frame may be appended.  Incomplete tails are discarded,
            # never padded or otherwise fabricated.
            if has_terminal_obs and has_complete_tail and self._cte_history is not None:
                next_image = self._build_robotwin_image_tensor(terminal_obs)[0].float()
                action_group = torch.stack(self._transition_actions).to(self.model.device)
                self._cte_history.append_transition(action_group, next_image)
                encoded = self._cte_history.forward()
                complete = encoded["effect_complete"][0]
                while self._observed_effect_count < complete.shape[0]:
                    effect_index = self._observed_effect_count
                    if bool(complete[effect_index]):
                        start = effect_index * self.cte.cfg.effect_window_transitions
                        self.lifecycle.observe_completed_effect(
                            encoded["phase"][0, start],
                            encoded["effect_post"][0, effect_index],
                            metadata={
                                "source_step": start * self.cte.cfg.transition_steps,
                                "terminal_attempt": True,
                                "attempt_success": bool(success),
                                "terminal_observation_used": True,
                            },
                        )
                    self._observed_effect_count += 1

            self._transition_actions.clear()
            # Commit effects observed earlier in this attempt even when the
            # terminal tail was incomplete or no reliable terminal frame was
            # available.
            if self.lifecycle.pim.task_cluster is not None:
                self.lifecycle.end_attempt()
            self._attempt_finalized = True
        except Exception:
            # Keep the guard unset so callers may retry after an operational
            # failure; do not claim the attempt was finalized prematurely.
            raise


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    cfg = _compose_sim_cfg(
        sim_cfg_path=sim_cfg_path,
        sim_cfg_name=sim_cfg_name,
        sim_task=sim_task,
    )

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    if device == "cpu" and model_dtype in {torch.float16, torch.bfloat16}:
        # Several Wan/attention kernels are not implemented for reduced
        # precision CPU execution.  Keep the explicit CPU fallback usable for
        # smoke tests; production RoboTwin runs should use CUDA.
        logger.warning("Using float32 for CPU evaluation because mixed precision=%s is CUDA-oriented.", mixed_precision)
        model_dtype = torch.float32

    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = eval_horizon if eval_horizon is not None else int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 8))

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps))

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("seed"))
    text_cfg_scale = float(usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0)))
    negative_prompt = str(usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", "")))
    rand_device = str(usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu")))
    tiled = _parse_bool(usr_args.get("tiled", cfg.EVALUATION.get("tiled", False)))
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
    )
    zeva_mode = str(usr_args.get("zeva_mode", cfg.EVALUATION.get("zeva_mode", "base")))
    cte_checkpoint = usr_args.get("cte_checkpoint", cfg.EVALUATION.get("cte_checkpoint"))
    addon_checkpoint = usr_args.get("addon_checkpoint", cfg.EVALUATION.get("addon_checkpoint"))
    pim_top_k = int(usr_args.get("pim_top_k", cfg.EVALUATION.get("pim_top_k", 4)))
    max_attempts = int(usr_args.get("max_attempts", cfg.EVALUATION.get("max_attempts", 1)))
    if zeva_mode != "base" and _parse_bool(usr_args.get("skip_get_obs_within_replan", cfg.EVALUATION.get("skip_get_obs_within_replan", False))):
        raise ValueError(
            "Zeva evaluation requires skip_get_obs_within_replan=false so every executed "
            "action can be paired with an observed after-frame"
        )

    policy = WorldActionRobotWinPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        timing_enabled=timing_enabled,
        num_video_frames=(int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1,
        video_size=tuple(int(v) for v in cfg.data.train.video_size),
        zeva_mode=zeva_mode,
        cte_checkpoint=None if _is_none_like(cte_checkpoint) else str(cte_checkpoint),
        addon_checkpoint=None if _is_none_like(addon_checkpoint) else str(addon_checkpoint),
        pim_top_k=pim_top_k,
        max_attempts=max_attempts,
    )
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()
