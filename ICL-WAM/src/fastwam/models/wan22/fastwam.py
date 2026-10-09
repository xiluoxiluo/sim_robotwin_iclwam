from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastwam.utils.logging_config import get_logger
from fastwam.zeva.causal_prompt import task_tokens_from_context
from fastwam.zeva.behavior_prefix_adapter import (
    ExactZevaPolicyInjectionAdapter,
    gaussian_prior_nll,
)

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


class FastWAM(torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        compile_training_denoise: bool = False,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.compile_training_denoise = bool(compile_training_denoise)
        self.mot.compile_training_layers = self.compile_training_denoise

        # Optional Zeva addon.  It is deliberately outside the frozen FastWAM
        # experts and is only populated by the Zeva runtime configuration.
        self.zeva_enabled = False
        self.zeva_prompt_encoder = None
        self.zeva_behavior_prefix_adapter = None
        self.zeva_injection_mode = "memory_residual"
        self.zeva_training_stage = "policy_injection"
        self.zeva_mode = "base"

        self.to(self.device)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = False,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        compile_training_denoise: bool = False,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAM.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            compile_training_denoise=compile_training_denoise,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    @torch.no_grad()
    def extract_task_context_readout(
        self, input_image: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Clean initial-image/text readout, with no action, proprio, or addon.

        Mirrors Zeva's mean of final vision hidden states at diffusion time
        zero. FastWAM's video MoT stream supplies the corresponding states.
        """
        if input_image.ndim != 4 or input_image.shape[1] != 3:
            raise ValueError("initial image must be [B,3,H,W]")
        if context.ndim != 3 or context.shape[0] != input_image.shape[0] or context_mask.shape != context.shape[:2]:
            raise ValueError("initial readout requires aligned image/context batches")
        if self.video_expert.training:
            raise ValueError("initial task-context readout requires the frozen policy in eval mode")
        context = context.to(device=self.device, dtype=self.torch_dtype)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        image = input_image.to(device=self.device, dtype=self.torch_dtype).unsqueeze(2)
        latent = self.vae.model.encode(image, self.vae.scale).clone()
        timestep = torch.zeros(input_image.shape[0], device=self.device, dtype=latent.dtype)
        (
            tokens, _time, t_mod, video_context, video_mask, freqs,
            _frames, _height, _width, tokens_per_frame,
        ) = self.video_expert.prepare(
            x=latent, timestep=timestep, context=context, context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)),
        )
        mask = self._build_mot_attention_mask(
            video_seq_len=tokens.shape[1], action_seq_len=0,
            video_tokens_per_frame=tokens_per_frame, device=self.device,
        )
        _keys, _values, hidden = self.mot.prefill_video_cache_tensor(
            video_tokens=tokens, video_freqs=freqs, video_t_mod=t_mod,
            video_context=video_context, video_context_mask=video_mask,
            video_attention_mask=mask, return_hidden=True,
        )
        return hidden.float().mean(dim=1).detach()

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    def _prepend_exact_zeva_behavior_slot(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        causal_prompt: torch.Tensor,
        pim_mask: torch.Tensor,
        task_tokens: torch.Tensor,
        *,
        enable_pim: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Insert the dedicated behavior slot for the exact Zeva adapter."""
        if getattr(self, "zeva_injection_mode", "memory_residual") != "exact_zeva":
            return context, context_mask
        adapter = self.zeva_behavior_prefix_adapter
        if adapter is None:
            raise RuntimeError("exact Zeva injection requires an attached adapter")
        return adapter.prepend_behavior_prefix_slot(
            context=context,
            context_mask=context_mask,
            causal_prompt=causal_prompt,
            pim_mask=pim_mask,
            task_tokens=task_tokens,
            enable_pim=enable_pim,
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if tiled:
            raise NotImplementedError("Batched VAE encoding does not support tiled encoding.")
        if not hasattr(self, "_vae_encode_compiled"):
            self._vae_encode_compiled = torch.compile(
                self.vae.model.encode,
                backend="cudagraphs",
                fullgraph=True,
            )
        return self._vae_encode_compiled(
            video_tensor.to(self.device),
            self.vae.scale,
        ).clone()

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        if tiled:
            raise NotImplementedError("Batched VAE image encoding does not support tiled encoding.")
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        return self.vae.model.encode(image.unsqueeze(0), self.vae.scale)

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for FastWAM training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        
        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)
        context = sample.get("context")
        context_mask = sample.get("context_mask")
        if context is None and context_mask is None:
            prompt = sample.get("prompt")
            if prompt is None:
                raise ValueError("FastWAM training requires `context/context_mask` or `prompt`.")
            context, context_mask = self.encode_prompt(prompt)
        elif context is None or context_mask is None:
            raise ValueError("`context` and `context_mask` must both exist when either is provided.")

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> first-frame video only
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def _joint_denoise_core(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        action_condition: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the tensor-only video/action core shared by training and inference."""
        (
            video_tokens,
            t_video,
            t_mod_video,
            context_video,
            context_mask_video,
            freqs_video,
            f_video,
            h_video,
            w_video,
            _tokens_per_frame,
        ) = self.video_expert.prepare(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action_condition,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        (
            action_tokens,
            _t_action,
            t_mod_action,
            context_action,
            context_mask_action,
            freqs_action,
        ) = self.action_expert.prepare(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        video_tokens, action_tokens = self.mot.forward_joint_core(
            video_tokens=video_tokens,
            action_tokens=action_tokens,
            video_freqs=freqs_video,
            action_freqs=freqs_action,
            video_t_mod=t_mod_video,
            action_t_mod=t_mod_action,
            video_context=context_video,
            video_context_mask=context_mask_video,
            action_context=context_action,
            action_context_mask=context_mask_action,
            attention_mask=attention_mask,
        )
        return (
            self.video_expert.post(video_tokens, t_video, f_video, h_video, w_video),
            self.action_expert.post(action_tokens),
        )

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        patch_t, patch_h, patch_w = (int(size) for size in self.video_expert.patch_size)
        latent_t, latent_h, latent_w = latents.shape[-3:]
        tokens_per_frame = (latent_h // patch_h) * (latent_w // patch_w)
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=(latent_t // patch_t) * tokens_per_frame,
            action_seq_len=noisy_action.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=latents.device,
        )
        pred_video, pred_action = self._joint_denoise_core(
            latents_video=latents,
            latents_action=noisy_action,
            timestep_video=timestep_video,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            attention_mask=attention_mask,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            action_condition=action,
        )

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2) # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patch_t, patch_h, patch_w = (int(size) for size in self.video_expert.patch_size)
        latent_t, latent_h, latent_w = latents_video.shape[-3:]
        tokens_per_frame = (latent_h // patch_h) * (latent_w // patch_w)
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=(latent_t // patch_t) * tokens_per_frame,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=latents_video.device,
        )
        return self._joint_denoise_core(
            latents_video=latents_video,
            latents_action=latents_action,
            timestep_video=timestep_video,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            attention_mask=attention_mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            action_condition=gt_action,
        )

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    def _denoise_action_with_video_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_cache_k: list[torch.Tensor],
        video_cache_v: list[torch.Tensor],
        action_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        (
            action_tokens,
            _t,
            action_t_mod,
            action_context,
            action_context_mask,
            action_freqs,
        ) = self.action_expert.prepare(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache_tensor(
            action_tokens=action_tokens,
            action_freqs=action_freqs,
            action_t_mod=action_t_mod,
            action_context=action_context,
            action_context_mask=action_context_mask,
            video_cache_k=video_cache_k,
            video_cache_v=video_cache_v,
            action_attention_mask=action_attention_mask,
        )
        return self.action_expert.post(action_tokens)

    def attach_zeva_addon(self, causal_prompt_encoder, behavior_prefix_adapter):
        """Attach the trainable Zeva modules without changing base FastWAM."""
        expected_hidden = int(getattr(self.action_expert, "hidden_dim", behavior_prefix_adapter.config.action_hidden_dim))
        actual_hidden = int(behavior_prefix_adapter.config.action_hidden_dim)
        if expected_hidden != actual_hidden:
            raise ValueError(
                f"Zeva adapter action_hidden_dim={actual_hidden} must match ActionDiT hidden_dim={expected_hidden}"
            )
        prompt_hidden = int(causal_prompt_encoder.config.hidden_dim)
        if int(behavior_prefix_adapter.config.memory_dim) != prompt_hidden:
            raise ValueError(
                f"Zeva adapter memory_dim={behavior_prefix_adapter.config.memory_dim} must match prompt hidden_dim={prompt_hidden}"
            )
        if int(behavior_prefix_adapter.config.action_horizon) != 32:
            raise ValueError("Zeva V1 BehaviorPrefixAdapter action_horizon must be 32")
        if int(self.action_expert.action_dim) != 14:
            raise ValueError("Zeva V1 requires a 14-dimensional FastWAM action expert")
        injection_mode = (
            "exact_zeva"
            if isinstance(behavior_prefix_adapter, ExactZevaPolicyInjectionAdapter)
            else "memory_residual"
        )
        if injection_mode == "exact_zeva":
            prompt_cfg = causal_prompt_encoder.config
            adapter_cfg = behavior_prefix_adapter.config
            dimension_pairs = (
                ("global_dim", adapter_cfg.global_dim, prompt_cfg.global_dim),
                ("phase_dim", adapter_cfg.phase_dim, prompt_cfg.phase_dim),
                ("effect_dim", adapter_cfg.effect_dim, prompt_cfg.effect_dim),
                ("effect_history_length", adapter_cfg.effect_history_length, prompt_cfg.brief_length),
            )
            for name, adapter_value, prompt_value in dimension_pairs:
                if int(adapter_value) != int(prompt_value):
                    raise ValueError(
                        f"exact Zeva {name} must match between policy adapter and "
                        f"CausalPromptEncoder: {adapter_value} vs {prompt_value}"
                    )
            if int(adapter_cfg.context_dim) != int(self.text_dim):
                raise ValueError(
                    "exact Zeva prefix context_dim must match FastWAM text_dim: "
                    f"{adapter_cfg.context_dim} vs {self.text_dim}"
                )
            if int(adapter_cfg.action_dim) != int(self.action_expert.action_dim):
                raise ValueError("exact Zeva prior action_dim must match FastWAM action_dim")
        # Make the Zeva contract safe even when the caller does not go through
        # Wan22Trainer: the pretrained FastWAM path is frozen before the addon
        # modules are attached, while the addon remains explicitly trainable.
        self.requires_grad_(False)
        self.zeva_prompt_encoder = causal_prompt_encoder.to(device=self.device, dtype=self.torch_dtype)
        self.zeva_behavior_prefix_adapter = behavior_prefix_adapter.to(device=self.device, dtype=self.torch_dtype)
        self.zeva_prompt_encoder.requires_grad_(True)
        self.zeva_behavior_prefix_adapter.requires_grad_(True)
        self.zeva_injection_mode = injection_mode
        self.zeva_enabled = True
        self.zeva_mode = "pim_on"
        return self

    def set_zeva_training_stage(self, stage: str):
        stage = str(stage)
        if stage not in {"policy_injection", "pim_adapter"}:
            raise ValueError(
                "Zeva training stage must be 'policy_injection' or 'pim_adapter'"
            )
        self.zeva_training_stage = stage
        return self

    def zeva_trainable_parameters(self):
        if not self.zeva_enabled or self.zeva_prompt_encoder is None or self.zeva_behavior_prefix_adapter is None:
            return []
        adapter = self.zeva_behavior_prefix_adapter
        if self.zeva_training_stage == "policy_injection":
            return list(adapter.policy_injection_parameters())
        if self.zeva_training_stage == "pim_adapter":
            return (
                list(self.zeva_prompt_encoder.parameters())
                + list(adapter.pim_adapter_parameters())
            )
        raise RuntimeError(
            f"unsupported Zeva training stage: {self.zeva_training_stage}"
        )

    def configure_zeva_trainable_state(self) -> None:
        if not self.zeva_enabled:
            raise RuntimeError("Zeva addon is not attached")
        self.eval()
        self.requires_grad_(False)
        adapter = self.zeva_behavior_prefix_adapter
        stage = self.zeva_training_stage
        if stage == "policy_injection":
            adapter.prior.train()
            adapter.prior.requires_grad_(True)
            adapter.action_prior_adapter.train()
            adapter.action_prior_adapter.requires_grad_(True)
            adapter.behavior_global_projector.train()
            adapter.behavior_global_projector.requires_grad_(True)
            return
        if stage == "pim_adapter":
            self.zeva_prompt_encoder.train()
            self.zeva_prompt_encoder.requires_grad_(True)
            adapter.prefix_project.train()
            adapter.prefix_project.requires_grad_(True)
            adapter.pim_gate.requires_grad_(True)
            return
        raise RuntimeError(f"unsupported Zeva training stage: {stage}")

    def _denoise_action_with_video_cache_zeva(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_cache_k: list[torch.Tensor],
        video_cache_v: list[torch.Tensor],
        action_attention_mask: torch.Tensor,
        causal_prompt: torch.Tensor,
        pim_mask: torch.Tensor,
        zeva_action_residual: Optional[torch.Tensor] = None,
        gate_override: Optional[float] = None,
        enable_prefix_injection: bool = True,
        enable_action_prior: bool = True,
        debug: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if self.zeva_behavior_prefix_adapter is None:
            raise RuntimeError("Zeva addon is not attached to this FastWAM instance")
        (
            action_tokens,
            _t,
            action_t_mod,
            action_context,
            action_context_mask,
            action_freqs,
        ) = self.action_expert.prepare(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        injection_mode = getattr(self, "zeva_injection_mode", "memory_residual")
        if injection_mode == "exact_zeva":
            if not enable_action_prior:
                gated_residual = torch.zeros_like(action_tokens)
            elif zeva_action_residual is None:
                raise ValueError("exact Zeva injection requires an action-prior residual")
            else:
                gated_residual = zeva_action_residual.to(
                    device=action_tokens.device, dtype=action_tokens.dtype
                )
        elif not enable_prefix_injection:
            gated_residual = torch.zeros_like(action_tokens)
        elif gate_override is None and self.zeva_behavior_prefix_adapter.training:
            memory_tokens = causal_prompt[:, None]
            memory_mask = pim_mask.any(dim=-1, keepdim=True)
            gated_residual = self.zeva_behavior_prefix_adapter.gated(
                memory_tokens, memory_mask, action_horizon=latents_action.shape[1]
            )
        else:
            memory_tokens = causal_prompt[:, None]
            memory_mask = pim_mask.any(dim=-1, keepdim=True)
            residual = self.zeva_behavior_prefix_adapter(
                memory_tokens, memory_mask, action_horizon=latents_action.shape[1]
            )
            gate = torch.tanh(self.zeva_behavior_prefix_adapter.pim_gate)
            if gate_override is not None:
                gate = gate.new_tensor(float(gate_override))
            gated_residual = gate * residual
        if injection_mode != "exact_zeva":
            has_pim = pim_mask.any(dim=-1).to(
                device=gated_residual.device, dtype=gated_residual.dtype
            ).view(-1, 1, 1)
            gated_residual = gated_residual * has_pim
        if gated_residual.shape != action_tokens.shape:
            raise ValueError(
                "Zeva action-prior residual must match prepared action hidden shape: "
                f"residual={tuple(gated_residual.shape)}, action={tuple(action_tokens.shape)}"
            )
        if debug is not None:
            base_norm = action_tokens.detach().float().norm(dim=-1).mean()
            delta_norm = gated_residual.detach().float().norm(dim=-1).mean()
            conditioned = action_tokens + gated_residual.to(dtype=action_tokens.dtype)
            debug.update(
                {
                    "base_action_hidden_norm": base_norm,
                    "memory_delta_hidden_norm": delta_norm,
                    "conditioned_action_hidden_norm": conditioned.detach().float().norm(dim=-1).mean(),
                    "memory_residual_ratio": delta_norm / base_norm.clamp_min(1.0e-8),
                }
            )
            action_tokens = conditioned
        else:
            action_tokens = action_tokens + gated_residual.to(dtype=action_tokens.dtype)
        action_tokens = self.mot.forward_action_with_video_cache_tensor(
            action_tokens=action_tokens,
            action_freqs=action_freqs,
            action_t_mod=action_t_mod,
            action_context=action_context,
            action_context_mask=action_context_mask,
            video_cache_k=video_cache_k,
            video_cache_v=video_cache_v,
            action_attention_mask=action_attention_mask,
        )
        return self.action_expert.post(action_tokens)

    def _validate_zeva_prompt(self, causal_prompt, pim_mask, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            not self.zeva_enabled
            or self.zeva_prompt_encoder is None
            or self.zeva_behavior_prefix_adapter is None
        ):
            raise RuntimeError("Zeva addon is not attached to this FastWAM instance")
        prompt_dim = int(self.zeva_prompt_encoder.config.hidden_dim)
        if causal_prompt.ndim != 2 or causal_prompt.shape != (batch_size, prompt_dim):
            raise ValueError(f"causal_prompt must be [B,{prompt_dim}]")
        if pim_mask.ndim != 2 or pim_mask.shape[0] != batch_size:
            raise ValueError("pim_mask must be [B,K]")
        return causal_prompt.to(self.device), pim_mask.to(self.device, dtype=torch.bool)

    def forward_zeva_action_train(
        self,
        input_image: torch.Tensor,
        clean_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        causal_prompt: torch.Tensor,
        pim_mask: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        action_valid: Optional[torch.Tensor] = None,
        zeva_action_residual: Optional[torch.Tensor] = None,
        zeva_prior_mean: Optional[torch.Tensor] = None,
        zeva_prior_std: Optional[torch.Tensor] = None,
        zeva_task_tokens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Train only the Zeva residual against FastWAM action flow matching."""
        if input_image.ndim != 4 or input_image.shape[1] != 3:
            raise ValueError("input_image must be [B,3,H,W]")
        if clean_action.ndim != 3 or clean_action.shape[0] != input_image.shape[0]:
            raise ValueError("clean_action must be [B,T,action_dim]")
        if clean_action.shape[1] != 32 or clean_action.shape[2] != self.action_expert.action_dim:
            raise ValueError("Zeva V1 requires clean_action [B,32,14]")
        causal_prompt, pim_mask = self._validate_zeva_prompt(
            causal_prompt, pim_mask, clean_action.shape[0]
        )
        context = context.to(self.device, dtype=self.torch_dtype)
        context_mask = context_mask.to(self.device, dtype=torch.bool)
        if (
            getattr(self, "zeva_injection_mode", "memory_residual") == "exact_zeva"
            and zeva_task_tokens is None
        ):
            raise ValueError(
                "exact Zeva action training requires explicit static task-context tokens"
            )
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("proprio is required when FastWAM proprio_dim is enabled")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(1)
            if proprio.ndim != 3 or proprio.shape[0] != clean_action.shape[0] or proprio.shape[-1] != self.proprio_dim:
                raise ValueError(
                    f"proprio must be [B,T,{self.proprio_dim}] for Zeva action training"
                )
            context, context_mask = self._append_proprio_to_context(
                context, context_mask, proprio[:, 0, :].to(self.device, dtype=self.torch_dtype)
            )
        if getattr(self, "zeva_injection_mode", "memory_residual") == "exact_zeva":
            context, context_mask = self._prepend_exact_zeva_behavior_slot(
                context=context,
                context_mask=context_mask,
                causal_prompt=causal_prompt,
                pim_mask=pim_mask,
                task_tokens=zeva_task_tokens,
                enable_pim=(self.zeva_training_stage == "pim_adapter"),
            )
        clean_action = clean_action.to(self.device, dtype=self.torch_dtype)
        noise = torch.randn_like(clean_action) if noise is None else noise.to(clean_action)
        timestep = self.train_action_scheduler.sample_training_t(
            batch_size=clean_action.shape[0], device=self.device, dtype=clean_action.dtype
        ) if timestep is None else timestep.to(self.device, dtype=clean_action.dtype)
        noisy_action = self.train_action_scheduler.add_noise(clean_action, noise, timestep)
        target_action = self.train_action_scheduler.training_target(clean_action, noise, timestep)

        # First-frame VAE and video KV construction are frozen, but the action
        # path below remains grad-enabled so gradients reach the addon.
        with torch.no_grad():
            image_video = input_image.to(self.device, dtype=self.torch_dtype).unsqueeze(2)
            # Use the same one-frame VAE operation as inference, but keep it
            # uncompiled here: Stage 2 supports batched CPU/smoke execution
            # and the compiled full-video path is CUDA-only in FastWAM.
            first_frame_latents = self.vae.model.encode(image_video, self.vae.scale).clone()
            timestep_video = torch.zeros((clean_action.shape[0],), device=self.device, dtype=first_frame_latents.dtype)
            (
                video_tokens, _tv, video_t_mod, video_context, video_context_mask,
                video_freqs, _fv, _hv, _wv, tokens_per_frame,
            ) = self.video_expert.prepare(
                x=first_frame_latents, timestep=timestep_video, context=context,
                context_mask=context_mask, action=None,
                fuse_vae_embedding_in_latents=bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)),
            )
            attention_mask = self._build_mot_attention_mask(
                video_seq_len=video_tokens.shape[1], action_seq_len=clean_action.shape[1],
                video_tokens_per_frame=tokens_per_frame, device=self.device,
            )
            video_cache_k, video_cache_v = self.mot.prefill_video_cache_tensor(
                video_tokens=video_tokens, video_freqs=video_freqs, video_t_mod=video_t_mod,
                video_context=video_context, video_context_mask=video_context_mask,
                video_attention_mask=attention_mask[: video_tokens.shape[1], : video_tokens.shape[1]],
            )
        debug_metrics: dict[str, torch.Tensor] = {}
        pred_action = self._denoise_action_with_video_cache_zeva(
            latents_action=noisy_action, timestep_action=timestep, context=context,
            context_mask=context_mask, video_cache_k=video_cache_k, video_cache_v=video_cache_v,
            action_attention_mask=attention_mask[video_tokens.shape[1]:, :],
            causal_prompt=causal_prompt, pim_mask=pim_mask,
            zeva_action_residual=zeva_action_residual,
            debug=debug_metrics,
        )
        error = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=-1)
        if action_valid is not None:
            valid = action_valid.to(self.device, dtype=error.dtype)
            if valid.shape != error.shape:
                raise ValueError("action_valid must be [B,32]")
            valid_sum = valid.sum(dim=1).clamp_min(1.0)
            loss_per_sample = (error * valid).sum(dim=1) / valid_sum
        else:
            loss_per_sample = error.mean(dim=1)
        action_weight = self.train_action_scheduler.training_weight(timestep).to(
            device=loss_per_sample.device, dtype=loss_per_sample.dtype
        )
        loss = (loss_per_sample * action_weight).mean()
        if (
            getattr(self, "zeva_injection_mode", "memory_residual") == "exact_zeva"
            and self.zeva_training_stage == "policy_injection"
        ):
            if zeva_prior_mean is None or zeva_prior_std is None:
                raise ValueError("exact Zeva training requires action-prior outputs")
            prior_loss = gaussian_prior_nll(
                clean_action.float(), zeva_prior_mean.float(), zeva_prior_std.float(), action_valid
            )
            loss = loss + prior_loss * float(self.zeva_behavior_prefix_adapter.config.prior_loss_weight)
            prior_metrics = {"behavior_prior_nll": float(prior_loss.detach().cpu())}
            residual = zeva_action_residual
            gate = float(torch.tanh(self.zeva_behavior_prefix_adapter.pim_gate.detach()).cpu())
        else:
            gate = float(torch.tanh(self.zeva_behavior_prefix_adapter.pim_gate.detach()).cpu())
            if getattr(self, "zeva_injection_mode", "memory_residual") == "exact_zeva":
                residual = zeva_action_residual
            else:
                residual = self.zeva_behavior_prefix_adapter(
                    causal_prompt[:, None],
                    pim_mask.any(dim=-1, keepdim=True),
                    action_horizon=clean_action.shape[1],
                )
            prior_metrics = {}
        residual_norm = float(residual.detach().float().norm(dim=-1).mean().cpu())
        metrics = {
            "loss_action": float(loss.detach()),
            "gate": gate,
            "behavior_residual_norm": residual_norm,
        }
        metrics.update(prior_metrics)
        for key, value in debug_metrics.items():
            metrics[key] = float(value.detach().cpu())
        # Keep stable metric names for rollout/trainer dashboards while
        # retaining the short keys used by existing callers.
        metrics.update({
            "memory/gate": metrics["gate"],
            "memory/base_action_hidden_norm": metrics.get("base_action_hidden_norm", 0.0),
            "memory/memory_delta_hidden_norm": metrics.get("memory_delta_hidden_norm", 0.0),
            "memory/conditioned_action_hidden_norm": metrics.get("conditioned_action_hidden_norm", 0.0),
            "memory/memory_residual_ratio": metrics.get("memory_residual_ratio", 0.0),
        })
        return loss, metrics

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        """Legacy dictionary-cache path retained for the optional IDM variant."""
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
        compile_action_infer: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
                compile_action_infer=compile_action_infer,
            )["action"]
        
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        patch_t, patch_h, patch_w = (int(size) for size in self.video_expert.patch_size)
        tokens_per_frame = (latent_h // patch_h) * (latent_w // patch_w)
        joint_attention_mask = self._build_mot_attention_mask(
            video_seq_len=(latent_t // patch_t) * tokens_per_frame,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=self.device,
        )
        if compile_action_infer:
            if action is not None:
                raise ValueError(
                    "`compile_action_infer=True` does not support action conditioning in `infer_joint`."
                )
            if not hasattr(self, "_joint_denoise_core_compiled_inference"):
                self._joint_denoise_core_compiled_inference = torch.compile(
                    self._joint_denoise_core,
                    mode="reduce-overhead",
                    fullgraph=True,
                )
            joint_denoise_core = self._joint_denoise_core_compiled_inference
        else:
            joint_denoise_core = self._joint_denoise_core

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            if compile_action_infer:
                torch.compiler.cudagraph_mark_step_begin()
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = joint_denoise_core(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                attention_mask=joint_attention_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                action_condition=action,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        compile_action_infer: bool = False,
        causal_prompt: Optional[torch.Tensor] = None,
        pim_mask: Optional[torch.Tensor] = None,
        zeva_action_residual: Optional[torch.Tensor] = None,
        zeva_task_tokens: Optional[torch.Tensor] = None,
        zeva_mode: str = "base",
        enable_prefix_injection: Optional[bool] = None,
        enable_action_prior: Optional[bool] = None,
    ) -> dict[str, Any]:
        self.eval()
        if zeva_mode not in {"base", "zeva_stage2", "pim_shadow", "pim_on"}:
            raise ValueError(
                "zeva_mode must be one of base, zeva_stage2, pim_shadow, pim_on"
            )
        if enable_prefix_injection is None:
            enable_prefix_injection = zeva_mode in {"zeva_stage2", "pim_on"}
        if enable_action_prior is None:
            enable_action_prior = zeva_mode in {"zeva_stage2", "pim_on"}
        if zeva_mode == "base":
            enable_prefix_injection = False
            enable_action_prior = False
        if zeva_mode != "base":
            if causal_prompt is None or pim_mask is None:
                raise ValueError("Zeva modes require causal_prompt and pim_mask")
            causal_prompt, pim_mask = self._validate_zeva_prompt(
                causal_prompt, pim_mask, 1
            )
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if (
            zeva_mode != "base"
            and getattr(self, "zeva_injection_mode", "memory_residual") == "exact_zeva"
            and zeva_task_tokens is None
        ):
            raise ValueError(
                "exact Zeva inference requires explicit static task-context tokens"
            )
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
        if (
            zeva_mode != "base"
            and getattr(self, "zeva_injection_mode", "memory_residual") == "exact_zeva"
            and enable_prefix_injection
        ):
            context, context_mask = self._prepend_exact_zeva_behavior_slot(
                context=context,
                context_mask=context_mask,
                causal_prompt=causal_prompt,
                pim_mask=pim_mask,
                task_tokens=zeva_task_tokens,
                enable_pim=(zeva_mode == "pim_on"),
            )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        (
            video_tokens,
            _t_video,
            video_t_mod,
            video_context,
            video_context_mask,
            video_freqs,
            _f_video,
            _h_video,
            _w_video,
            tokens_per_frame,
        ) = self.video_expert.prepare(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_tokens.shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=video_tokens.device,
        )
        video_attention_mask = attention_mask[:video_seq_len, :video_seq_len]
        action_attention_mask = attention_mask[video_seq_len:, :]
        if compile_action_infer:
            if not hasattr(self, "_prefill_video_cache_compiled"):
                self._prefill_video_cache_compiled = torch.compile(
                    self.mot.prefill_video_cache_tensor,
                    mode="reduce-overhead",
                    fullgraph=True,
                )
            if not hasattr(self, "_denoise_action_with_video_cache_compiled"):
                self._denoise_action_with_video_cache_compiled = torch.compile(
                    self._denoise_action_with_video_cache,
                    mode="reduce-overhead",
                    fullgraph=True,
                )
            prefill_video_cache = self._prefill_video_cache_compiled
            denoise_action_with_video_cache = self._denoise_action_with_video_cache_compiled
        else:
            prefill_video_cache = self.mot.prefill_video_cache_tensor
            denoise_action_with_video_cache = self._denoise_action_with_video_cache
        if compile_action_infer:
            torch.compiler.cudagraph_mark_step_begin()
        video_cache_k, video_cache_v = prefill_video_cache(
            video_tokens=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_t_mod,
            video_context=video_context,
            video_context_mask=video_context_mask,
            video_attention_mask=video_attention_mask,
        )
        if compile_action_infer:
            # Inductor reduce-overhead may return graph-owned buffers that are overwritten on replay.
            video_cache_k = [cache.clone() for cache in video_cache_k]
            video_cache_v = [cache.clone() for cache in video_cache_v]

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            if compile_action_infer:
                torch.compiler.cudagraph_mark_step_begin()
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            if zeva_mode == "base":
                pred_action_posi = denoise_action_with_video_cache(
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    video_cache_k=video_cache_k,
                    video_cache_v=video_cache_v,
                    action_attention_mask=action_attention_mask,
                )
            else:
                pred_action_posi = self._denoise_action_with_video_cache_zeva(
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    video_cache_k=video_cache_k,
                    video_cache_v=video_cache_v,
                    action_attention_mask=action_attention_mask,
                    causal_prompt=causal_prompt,
                    pim_mask=pim_mask,
                    zeva_action_residual=zeva_action_residual,
                    enable_prefix_injection=enable_prefix_injection,
                    enable_action_prior=enable_action_prior,
                    gate_override=(
                        0.0
                        if (
                            zeva_mode != "pim_on"
                            and getattr(self, "zeva_injection_mode", "memory_residual")
                            != "exact_zeva"
                        )
                        else None
                    ),
                )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def infer_action_zeva(self, *, causal_prompt, pim_mask, zeva_mode: str = "pim_on", **kwargs):
        """Explicit Zeva inference entry point; base infer_action remains compatible."""
        return self.infer_action(
            causal_prompt=causal_prompt,
            pim_mask=pim_mask,
            zeva_mode=zeva_mode,
            **kwargs,
        )

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def save_zeva_addon_checkpoint(self, path, *, step: int = 0, config: Optional[dict] = None,
                                   base_checkpoint_sha256: str = "", cte_checkpoint_sha256: str = ""):
        if not self.zeva_enabled or self.zeva_prompt_encoder is None or self.zeva_behavior_prefix_adapter is None:
            raise RuntimeError("Cannot save a Zeva addon before attaching it")
        torch.save({
            "causal_prompt_encoder": self.zeva_prompt_encoder.state_dict(),
            "behavior_prefix_adapter": self.zeva_behavior_prefix_adapter.state_dict(),
            "pim_gate": self.zeva_behavior_prefix_adapter.pim_gate.detach().cpu(),
            "training_stage": self.zeva_training_stage,
            "step": int(step), "config": config or {},
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "cte_checkpoint_sha256": cte_checkpoint_sha256,
            "task_context_identity": getattr(self, "zeva_task_context_identity", None),
        }, path)

    def load_zeva_addon_checkpoint(self, path, *, base_checkpoint_sha256: Optional[str] = None,
                                   cte_checkpoint_sha256: Optional[str] = None,
                                   load_scope: str = "all"):
        if not self.zeva_enabled or self.zeva_prompt_encoder is None or self.zeva_behavior_prefix_adapter is None:
            raise RuntimeError("Cannot load a Zeva addon before attaching it")
        from fastwam.zeva.checkpoint import load_addon_checkpoint
        return load_addon_checkpoint(
            path, self.zeva_prompt_encoder, self.zeva_behavior_prefix_adapter,
            base_checkpoint_sha256=base_checkpoint_sha256,
            cte_checkpoint_sha256=cte_checkpoint_sha256,
            task_context_identity=getattr(self, "zeva_task_context_identity", None),
            load_scope=load_scope,
        )

    def zeva_parameter_report(self) -> dict[str, object]:
        trainable = [name for name, p in self.named_parameters() if p.requires_grad]
        frozen = [name for name, p in self.named_parameters() if not p.requires_grad]
        stage = self.zeva_training_stage
        if stage == "policy_injection":
            allowed = (
                "zeva_behavior_prefix_adapter.prior.",
                "zeva_behavior_prefix_adapter.action_prior_adapter.",
                "zeva_behavior_prefix_adapter.behavior_global_projector.",
            )
        elif stage == "pim_adapter":
            allowed = (
                "zeva_prompt_encoder.",
                "zeva_behavior_prefix_adapter.prefix_project.",
                "zeva_behavior_prefix_adapter.pim_gate",
            )
        else:
            raise RuntimeError(f"unsupported Zeva training stage: {stage}")
        invalid = [
            name
            for name in trainable
            if not any(name.startswith(prefix) for prefix in allowed)
        ]
        if invalid:
            raise AssertionError(
                f"Unexpected trainable Zeva parameters for stage={stage}: {invalid[:10]}"
            )
        return {
            "training_stage": stage,
            "trainable_names": trainable,
            "trainable_count": sum(self.get_parameter(n).numel() for n in trainable),
            "frozen_count": sum(self.get_parameter(n).numel() for n in frozen),
        }

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=False)
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        if args and isinstance(args[0], dict):
            sample = args[0]
            if sample.get("_training_mode", "base") == "zeva_stage2":
                return self._forward_zeva_stage2(sample)
        return self.training_loss(*args, **kwargs)

    def _forward_zeva_stage2(self, sample: dict[str, Any]):
        """Prepared-model forward route for Zeva Stage 2.

        Keeping prompt construction and the conditioned action loss behind the
        module's forward entry preserves DDP/DeepSpeed forward hooks. Frozen
        video/VAE work remains bounded by ``forward_zeva_action_train``'s
        no-grad region.
        """
        if not self.zeva_enabled or self.zeva_prompt_encoder is None:
            raise RuntimeError("Zeva Stage 2 forward requires an attached addon")
        stage = self.zeva_training_stage
        required = (
            "video",
            "action",
            "context",
            "context_mask",
            "phase",
            "bit_effects",
            "bit_mask",
        )
        if stage == "pim_adapter":
            required += ("pim_phases", "pim_effects", "pim_mask")
        missing = [key for key in required if key not in sample]
        if missing:
            raise KeyError(f"Zeva Stage 2 sample is missing fields: {missing}")
        video = sample["video"]
        if video.ndim != 5 or video.shape[2] < 1:
            raise ValueError("Zeva Stage 2 sample video must be [B,3,T,H,W]")
        context = sample["context"].to(self.device)
        context_mask = sample["context_mask"].to(self.device, dtype=torch.bool)
        task_dim = int(self.zeva_prompt_encoder.config.global_dim)
        task_mode = str(getattr(self, "zeva_task_context_mode", "static"))
        if sample.get("task_context") is None:
            if task_mode == "static":
                raise ValueError("formal Zeva training requires static task_context")
            if task_mode != "pooling":
                raise ValueError(f"unsupported Zeva task_context mode: {task_mode}")
            task_tokens = task_tokens_from_context(context, context_mask, task_dim)
        else:
            task_tokens = sample["task_context"].to(device=self.device)
        if task_tokens.shape != (context.shape[0], task_dim):
            raise ValueError("task_context must be [B, prompt.global_dim]")
        batch = context.shape[0]
        prompt_dim = int(self.zeva_prompt_encoder.config.hidden_dim)
        pim_k = int(self.zeva_prompt_encoder.config.persistent_length)
        if stage == "policy_injection":
            causal_prompt = torch.zeros(
                (batch, prompt_dim),
                device=self.device,
                dtype=self.torch_dtype,
            )
            pim_mask_for_policy = torch.zeros(
                (batch, pim_k),
                device=self.device,
                dtype=torch.bool,
            )
        elif stage == "pim_adapter":
            causal_prompt = self.zeva_prompt_encoder(
                task_tokens=task_tokens,
                current_phase=sample["phase"].to(self.device),
                bit_effects=sample["bit_effects"].to(self.device),
                bit_mask=sample["bit_mask"].to(self.device),
                pim_phases=sample["pim_phases"].to(self.device),
                pim_effects=sample["pim_effects"].to(self.device),
                pim_mask=sample["pim_mask"].to(self.device),
            )
            pim_mask_for_policy = sample["pim_mask"].to(
                self.device,
                dtype=torch.bool,
            )
        else:
            raise RuntimeError(f"unsupported Zeva training stage: {stage}")
        zeva_action_residual = None
        zeva_prior_mean = None
        zeva_prior_std = None
        if getattr(self, "zeva_injection_mode", "memory_residual") == "exact_zeva":
            adapter = self.zeva_behavior_prefix_adapter
            zeva_prior_mean, zeva_prior_std = adapter.prior(
                task_tokens,
                sample["phase"].to(self.device),
                sample["bit_effects"].to(self.device),
                sample["bit_mask"].to(self.device),
            )
            zeva_action_residual = adapter.action_prior_residual(
                zeva_prior_mean, training=True
            )
        loss, metrics = self.forward_zeva_action_train(
            input_image=video[:, :, 0],
            clean_action=sample["action"],
            context=context,
            context_mask=context_mask,
            causal_prompt=causal_prompt,
            pim_mask=pim_mask_for_policy,
            proprio=sample.get("proprio"),
            action_valid=sample.get("action_valid"),
            zeva_action_residual=zeva_action_residual,
            zeva_prior_mean=zeva_prior_mean,
            zeva_prior_std=zeva_prior_std,
            zeva_task_tokens=task_tokens,
        )
        metrics.update(
            {
                "memory/bit_count": float(sample["bit_mask"].to(dtype=torch.float32).sum(dim=-1).mean().detach().cpu()),
                "memory/pim_count": float(
                    sample.get(
                        "pim_mask",
                        torch.zeros((batch, pim_k), dtype=torch.bool),
                    )
                    .to(dtype=torch.float32)
                    .sum(dim=-1)
                    .mean()
                    .detach()
                    .cpu()
                ),
            }
        )
        return loss, metrics
