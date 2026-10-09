# SPDX-FileCopyrightText: Copyright (c) 2026 Z-Trans CORPORATION. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Explicit RGB-to-CTE input adapters.

The Zeva CTE itself is deliberately kept generic: it consumes a tensor with
shape ``[B,T,C,H,W]`` and does not know how the tensor was produced.

For the FastWAM / RoboTwin Zeva path, RGB observations are encoded by the
frozen Wan VAE before entering CTE. Keeping the VAE boundary outside CTE
makes the representation contract explicit and also allows the exact same
frozen VAE outputs to be precomputed into the Stage-1 latent cache.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


_VAE_IDENTITY_KEYS = (
    "model_id",
    "z_dim",
    "temporal_downsample_factor",
    "upsampling_factor",
)


def validate_vae_metadata(
    expected: dict[str, object],
    actual: dict[str, object],
) -> None:
    """Check semantic VAE identity while allowing paths to differ per host."""

    for key in _VAE_IDENTITY_KEYS:
        if key not in expected or key not in actual:
            raise ValueError(
                f"VAE metadata is missing required identity field {key!r}"
            )

        if str(expected[key]) != str(actual[key]):
            raise ValueError(
                f"VAE metadata mismatch for {key}: "
                f"expected {expected[key]!r}, "
                f"got {actual[key]!r}"
            )


class FastWAMCTELatentEncoder:
    """Encode RGB frames with an already-loaded frozen FastWAM/Wan VAE.

    Input
    -----
    Single frame batch:
        [B, 3, H, W]

    History batch:
        [B, T, 3, H, W]

    Output
    ------
    Single-frame latent:
        [B, C_latent, H_latent, W_latent]

    History latent:
        [B, T, C_latent, H_latent, W_latent]

    Notes
    -----
    The returned latent is float32. This preserves the original Stage-1 CTE
    training contract.

    The offline latent-cache builder may subsequently cast the returned values
    to BF16 and store their raw uint16 bit pattern. The training dataset then
    reconstructs BF16 values and converts them back to float32 before CTE.
    """

    def __init__(
        self,
        model: Any,
        *,
        resize: tuple[int, int] = (480, 832),
        expected_channels: int | None = None,
        input_range: str = "auto",
    ) -> None:
        self.model = model

        self.resize = tuple(
            int(value)
            for value in resize
        )

        self.expected_channels = (
            None
            if expected_channels is None
            else int(expected_channels)
        )

        self.input_range = str(input_range)

        if len(self.resize) != 2:
            raise ValueError(
                f"resize must contain exactly two dimensions, got {self.resize}"
            )

        if min(self.resize) < 1:
            raise ValueError(
                f"resize must contain positive dimensions, got {self.resize}"
            )

        if (
            self.expected_channels is not None
            and self.expected_channels < 1
        ):
            raise ValueError(
                "expected_channels must be positive"
            )

        if self.input_range not in {
            "auto",
            "zero_one",
            "minus_one_one",
            "uint8",
        }:
            raise ValueError(
                "input_range must be one of: "
                "auto, zero_one, minus_one_one, uint8"
            )

    # ------------------------------------------------------------------
    # RGB preprocessing
    # ------------------------------------------------------------------

    def _as_rgb_batch(
        self,
        rgb: Tensor,
    ) -> Tensor:
        """Normalize RGB input to float32 [-1, 1].

        Accepts:
            [3,H,W]
            [B,3,H,W]

        Supported source ranges:
            uint8        : [0,255]
            zero_one     : [0,1]
            minus_one_one: [-1,1]
            auto         : infer from values
        """

        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)

        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(
                "rgb must be [B,3,H,W] or [3,H,W], "
                f"got {tuple(rgb.shape)}"
            )

        # Explicit uint8 path.
        if (
            self.input_range == "uint8"
            or rgb.dtype == torch.uint8
        ):
            rgb = rgb.float()
            rgb = rgb / 127.5 - 1.0
            return rgb.clamp(-1.0, 1.0)

        # Already normalized to [-1, 1].
        if self.input_range == "minus_one_one":
            return rgb.float().clamp(-1.0, 1.0)

        # Explicit [0, 1].
        if self.input_range == "zero_one":
            rgb = rgb.float().clamp(0.0, 1.0)
            return rgb * 2.0 - 1.0

        # --------------------------------------------------------------
        # AUTO mode
        # --------------------------------------------------------------
        rgb = rgb.float()

        minimum = float(
            rgb.detach().amin().item()
        )
        maximum = float(
            rgb.detach().amax().item()
        )

        # Looks like [-1, 1].
        if minimum < -1.0e-4:
            if maximum > 1.0001:
                raise ValueError(
                    "RGB values are outside supported ranges. "
                    f"Observed min={minimum:.6f}, max={maximum:.6f}. "
                    "Expected uint8/[0,255], [0,1], or [-1,1]."
                )

            return rgb.clamp(
                -1.0,
                1.0,
            )

        # Looks like [0,255].
        if maximum > 1.0001:
            if maximum > 255.0001:
                raise ValueError(
                    "RGB values exceed expected uint8 range. "
                    f"Observed max={maximum:.6f}."
                )

            rgb = rgb / 127.5 - 1.0
            return rgb.clamp(
                -1.0,
                1.0,
            )

        # Otherwise treat it as [0,1].
        rgb = rgb.clamp(
            0.0,
            1.0,
        )

        return rgb * 2.0 - 1.0

    # ------------------------------------------------------------------
    # VAE model helpers
    # ------------------------------------------------------------------

    def _model_device_dtype(
        self,
    ) -> tuple[torch.device | None, torch.dtype | None]:
        """Return the underlying VAE parameter device/dtype when possible."""

        candidates: list[Any] = [
            self.model,
            getattr(self.model, "model", None),
            getattr(
                getattr(self.model, "vae", None),
                "model",
                None,
            ),
        ]

        for module in candidates:
            if not isinstance(
                module,
                torch.nn.Module,
            ):
                continue

            try:
                parameter = next(
                    module.parameters()
                )
            except StopIteration:
                continue

            return (
                parameter.device,
                parameter.dtype,
            )

        return None, None

    def _encode(
        self,
        normalized: Tensor,
    ) -> Tensor:
        """Run frozen Wan VAE on normalized RGB.

        ``normalized``:
            [B,3,H,W]

        Wan's VAE expects:
            [B,3,T,H,W]

        Therefore a singleton temporal axis is inserted. Since Wan VAE is
        causal, a one-frame RGB observation produces one latent time step.
        """

        if normalized.ndim != 4:
            raise ValueError(
                "normalized RGB must be [B,3,H,W], "
                f"got {tuple(normalized.shape)}"
            )

        if normalized.shape[1] != 3:
            raise ValueError(
                "normalized RGB must have 3 channels, "
                f"got {normalized.shape[1]}"
            )

        # --------------------------------------------------------------
        # Match VAE device / dtype.
        # --------------------------------------------------------------
        model_device, model_dtype = (
            self._model_device_dtype()
        )

        if model_device is not None:
            normalized = normalized.to(
                device=model_device,
                non_blocking=True,
            )

        if (
            model_dtype is not None
            and model_dtype.is_floating_point
        ):
            normalized = normalized.to(
                dtype=model_dtype,
            )

        # [B,3,H,W]
        #       ↓
        # [B,3,1,H,W]
        video = normalized.unsqueeze(2)

        # --------------------------------------------------------------
        # FastWAM / Wan wrappers supported by the repository.
        # --------------------------------------------------------------

        if (
            hasattr(self.model, "vae")
            and hasattr(self.model.vae, "model")
        ):
            latent = (
                self.model.vae.model.encode(
                    video,
                    self.model.vae.scale,
                )
            )

        elif hasattr(
            self.model,
            "_encode_video_latents",
        ):
            latent = (
                self.model._encode_video_latents(
                    video,
                    tiled=False,
                )
            )

        elif (
            hasattr(self.model, "model")
            and hasattr(
                self.model.model,
                "encode",
            )
        ):
            scale = getattr(
                self.model,
                "scale",
                None,
            )

            if scale is None:
                raise TypeError(
                    "VAE wrapper exposes model.encode(), "
                    "but does not expose normalization scale"
                )

            latent = self.model.model.encode(
                video,
                scale,
            )

        elif hasattr(
            self.model,
            "encode",
        ):
            latent = self.model.encode(
                video
            )

        elif hasattr(
            self.model,
            "_encode_input_image_latents_tensor",
        ):
            # Compatibility path for a FastWAM model exposing only
            # the single-image helper.
            outputs = []

            for frame in normalized:
                output = (
                    self.model
                    ._encode_input_image_latents_tensor(
                        frame,
                        tiled=False,
                    )
                )
                outputs.append(output)

            latent = torch.cat(
                outputs,
                dim=0,
            )

        else:
            raise TypeError(
                "model must expose one of: "
                "vae.model.encode(), "
                "_encode_video_latents(), "
                "model.encode(), "
                "encode(), or "
                "_encode_input_image_latents_tensor()"
            )

        # --------------------------------------------------------------
        # Normalize output shape.
        # --------------------------------------------------------------
        if not torch.is_tensor(latent):
            raise TypeError(
                "VAE encoder must return a torch.Tensor, "
                f"got {type(latent)!r}"
            )

        if latent.ndim != 5:
            raise ValueError(
                "VAE encoder must return "
                "[B,C,T,H,W], "
                f"got {tuple(latent.shape)}"
            )

        if latent.shape[2] != 1:
            raise ValueError(
                "single RGB frame must produce exactly "
                "one latent time step; "
                f"got latent shape {tuple(latent.shape)}"
            )

        if (
            latent.shape[0]
            != normalized.shape[0]
        ):
            raise ValueError(
                "VAE encoder changed the batch size: "
                f"input={normalized.shape[0]}, "
                f"output={latent.shape[0]}"
            )

        # [B,C,1,H,W]
        #       ↓
        # [B,C,H,W]
        latent = (
            latent[:, :, 0]
            .float()
            .contiguous()
        )

        if (
            self.expected_channels is not None
            and latent.shape[1]
            != self.expected_channels
        ):
            raise ValueError(
                "VAE latent channel mismatch: "
                f"expected {self.expected_channels}, "
                f"got {latent.shape[1]}"
            )

        return latent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        rgb: Tensor,
    ) -> Tensor:
        """Encode RGB observations into individual Wan-VAE latent frames.

        Input:
            [B,3,H,W]
            or
            [3,H,W]

        Output:
            [B,C_latent,H_latent,W_latent]
        """

        normalized = self._as_rgb_batch(
            rgb
        )

        # --------------------------------------------------------------
        # Important optimization:
        #
        # RoboTwin CTE currently already provides 384x320 observations and
        # cte_vae_input_size is also 384x320. Do NOT execute a useless
        # 384x320 -> 384x320 bilinear interpolation.
        # --------------------------------------------------------------
        if (
            tuple(normalized.shape[-2:])
            != self.resize
        ):
            normalized = F.interpolate(
                normalized,
                size=self.resize,
                mode="bilinear",
                align_corners=False,
            )

        return self._encode(
            normalized
        )

    @torch.no_grad()
    def encode_history(
        self,
        rgb_history: Tensor,
    ) -> Tensor:
        """Encode RGB histories into CTE Wan-VAE latents.

        Input:
            [B,T,3,H,W]

        Output:
            [B,T,C_latent,H_latent,W_latent]

        The temporal dimension is flattened into the image batch so all
        observations are sent through Wan VAE in one batched call.

        This preserves the original Zeva/FastWAM CTE convention in which
        each CTE observation boundary is represented by a causal one-frame
        Wan-VAE encoding.
        """

        if (
            rgb_history.ndim != 5
            or rgb_history.shape[2] != 3
        ):
            raise ValueError(
                "rgb_history must be "
                "[B,T,3,H,W], "
                f"got {tuple(rgb_history.shape)}"
            )

        batch = int(
            rgb_history.shape[0]
        )

        steps = int(
            rgb_history.shape[1]
        )

        if batch < 1 or steps < 1:
            raise ValueError(
                "rgb_history batch/time dimensions "
                "must be positive"
            )

        # [B,T,3,H,W]
        #       ↓
        # [B*T,3,H,W]
        flattened = (
            rgb_history
            .flatten(0, 1)
            .contiguous()
        )

        latent = self.encode(
            flattened
        )

        if latent.shape[0] != batch * steps:
            raise ValueError(
                "VAE history encoding changed "
                "the flattened batch size: "
                f"expected {batch * steps}, "
                f"got {latent.shape[0]}"
            )

        # [B*T,C,H',W']
        #        ↓
        # [B,T,C,H',W']
        return latent.reshape(
            batch,
            steps,
            *latent.shape[1:],
        )


def load_frozen_wan_vae(
    *,
    model_id: str,
    tokenizer_model_id: str,
    device: str,
    torch_dtype: torch.dtype = torch.float32,
    redirect_common_files: bool = True,
) -> tuple[Any, dict[str, object]]:
    """Load only the frozen Wan VAE through FastWAM's registered loader."""

    from fastwam.models.wan22.helpers.loader import (
        _load_registered_model,
        _resolve_configs,
    )

    (
        _dit_config,
        _text_config,
        vae_config,
        _tokenizer_config,
    ) = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )

    vae_config.download_if_necessary()

    vae = _load_registered_model(
        vae_config.path,
        "wan_video_vae",
        torch_dtype=torch_dtype,
        device=device,
    )

    vae.eval()
    vae.requires_grad_(False)

    metadata = {
        "model_id": str(
            model_id
        ),
        "vae_path": str(
            vae_config.path
        ),
        "z_dim": int(
            getattr(
                vae,
                "z_dim",
                getattr(
                    getattr(
                        vae,
                        "model",
                        None,
                    ),
                    "z_dim",
                    0,
                ),
            )
        ),
        "temporal_downsample_factor": int(
            getattr(
                vae,
                "temporal_downsample_factor",
                0,
            )
        ),
        "upsampling_factor": int(
            getattr(
                vae,
                "upsampling_factor",
                0,
            )
        ),
    }

    # Do not silently create a cache/checkpoint whose VAE identity is
    # incomplete.
    required_positive = (
        "z_dim",
        "temporal_downsample_factor",
        "upsampling_factor",
    )

    for key in required_positive:
        if int(metadata[key]) < 1:
            raise ValueError(
                f"loaded Wan VAE does not expose "
                f"a valid {key}: {metadata[key]!r}"
            )

    return (
        vae,
        metadata,
    )


def make_frame_encoder(
    input_type: str,
    *,
    model: Any | None = None,
    resize: tuple[int, int] = (480, 832),
    expected_channels: int | None = None,
) -> Callable[[Tensor], Tensor] | None:
    """Build the configured CTE input encoder."""

    input_type = str(
        input_type
    )

    if input_type == "rgb_frame":
        return None

    if input_type != "wan_vae_latent":
        raise ValueError(
            "input_type must be "
            "'rgb_frame' or 'wan_vae_latent'"
        )

    if model is None:
        raise ValueError(
            "wan_vae_latent input requires "
            "an explicit frozen Wan VAE model"
        )

    encoder = FastWAMCTELatentEncoder(
        model,
        resize=resize,
        expected_channels=expected_channels,
        input_range="minus_one_one",
    )

    return encoder.encode_history