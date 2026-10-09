import hashlib
import os
from typing import Optional
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
from contextlib import contextmanager

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from fastwam.utils.logging_config import get_logger
from fastwam.utils import misc, pytorch_utils
from fastwam.zeva.semantic_tasks import SemanticTaskMap
from accelerate import PartialState

logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


class RobotVideoDataset(torch.utils.data.Dataset):
    base_dataset_cls = BaseLerobotDataset

    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        use_text_embed_cache=True,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal",  # horizontal, vertical, robotwin, or None
        override_instruction: Optional[str] = None,
        tolerance_s: Optional[float] = None,
        video_backend: Optional[str] = None,
        # Zeva semantic identity.  Optional for ordinary FastWAM, required by
        # the repaired RoboTwin Zeva configs.
        semantic_task_map_path: Optional[str] = None,
        require_semantic_task_id: bool = False,
    ):
        self.dataset_dirs = list(dataset_dirs)
        self.lerobot_dataset = self.base_dataset_cls(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            image_subsample_stride=action_video_freq_ratio,
            tolerance_s=tolerance_s,
            video_backend=video_backend,
        )

        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio

        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.use_text_embed_cache = bool(use_text_embed_cache)
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction

        self.require_semantic_task_id = bool(require_semantic_task_id)
        self.semantic_task_map_path = (
            None
            if semantic_task_map_path in (None, "", "None", "null")
            else str(semantic_task_map_path)
        )
        self.semantic_task_map = (
            None
            if self.semantic_task_map_path is None
            else SemanticTaskMap.load(self.semantic_task_map_path)
        )
        if self.require_semantic_task_id and self.semantic_task_map is None:
            raise ValueError(
                "require_semantic_task_id=true but semantic_task_map_path is not set"
            )

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if self.lerobot_dataset.presample_images:
                processor.num_image_steps = len(self.video_sample_indices)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError(
                        "pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them."
                    )
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(
                        dataset_stats, os.path.join(work_dir, "dataset_stats.json")
                    )
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(
                        dataset_stats, os.path.join(work_dir, "dataset_stats.json")
                    )

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)

    @property
    def semantic_task_identity(self) -> dict | None:
        if self.semantic_task_map is None:
            return None
        return self.semantic_task_map.identity.to_dict()

    def resolve_semantic_task_id(
        self,
        *,
        episode_index: int | None,
        raw_task_index: int | str | None = None,
        strict: bool | None = None,
    ) -> str | int:
        """Return canonical RoboTwin task identity for Zeva.

        When no semantic map is configured we preserve the historical fallback
        for ordinary FastWAM/non-Zeva uses.  Repaired Zeva configs set
        ``require_semantic_task_id=true``, so a missing mapping is a hard error.
        """
        if strict is None:
            strict = self.require_semantic_task_id
        if self.semantic_task_map is not None:
            resolved = self.semantic_task_map.resolve(
                episode_index=episode_index,
                raw_task_index=raw_task_index,
                strict=bool(strict),
            )
            if resolved is not None:
                return resolved
        if strict:
            raise KeyError(
                f"missing semantic task for episode_index={episode_index}, raw_task_index={raw_task_index}"
            )
        if raw_task_index is not None:
            return raw_task_index
        return "unknown-task"

    def __len__(self):
        return len(self.lerobot_dataset)

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))

        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]
        num_cameras = 1
        if video.ndim == 5:
            if not self.lerobot_dataset.presample_images:
                video = video[:, self.video_sample_indices, :, :, :]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            if not self.lerobot_dataset.presample_images:
                video = video[self.video_sample_indices, :, :, :]
            T_video, C, H, W = video.shape
        if not self.lerobot_dataset.presample_images:
            image_is_pad = image_is_pad[self.video_sample_indices]

        video = video.view(num_cameras, T_video, C, H, W)
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            bottom = torch.cat([cam_left, cam_right], dim=-1)
            video = torch.cat([cam_top, bottom], dim=-2)
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)

        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)
        video = video.permute(1, 0, 2, 3)

        action = sample["action"]
        proprio = sample["proprio"][:-1, :]
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )

        task = sample["instruction"]
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "dataset_index": int(sample.get("idx", sample_idx)),
            "prompt": instruction,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }

        for key in ("episode_index", "frame_index", "index", "task_index"):
            if key in sample:
                value = sample[key]
                if torch.is_tensor(value):
                    value = value.flatten()[0].item() if value.numel() else 0
                elif isinstance(value, np.ndarray):
                    value = value.reshape(-1)[0].item() if value.size else 0
                data[key] = value

        if "episode_index" in data:
            episode_index = int(data["episode_index"])
            data["episode_id"] = f"{self.dataset_dirs[0]}::episode-{episode_index}"
            data["episode_step"] = int(data.get("frame_index", data.get("index", 0)))
        else:
            episode_index = None
            data["episode_id"] = f"{self.dataset_dirs[0]}::window-{data['dataset_index']}"
            data["episode_step"] = 0

        raw_task_index = data.get("task_index")
        data["raw_task_index"] = raw_task_index
        data["raw_task_name"] = str(task)

        semantic_task_id = self.resolve_semantic_task_id(
            episode_index=episode_index,
            raw_task_index=raw_task_index if raw_task_index is not None else str(task),
        )
        data["semantic_task_id"] = semantic_task_id
        data["task_id"] = semantic_task_id
        # task_name is the semantic task identity for Zeva bookkeeping; the
        # original natural-language instruction remains in raw_task_name/prompt.
        data["task_name"] = str(semantic_task_id)

        if self.use_text_embed_cache:
            context, context_mask = self._get_cached_text_context(instruction)
            context[~context_mask] = 0.0
            data["context"] = context
            data["context_mask"] = torch.ones_like(context_mask)
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(
            cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt"
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )
        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
