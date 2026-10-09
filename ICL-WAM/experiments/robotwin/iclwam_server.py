"""Stateful ICL-WAM HTTP inference for RoboTwin.

The FastWAM /act wire format is retained, but each reply contains ONE qpos
action. Call again after executing it: CTE must observe real transitions.
See docs/iclwam_robotwin_server.md for startup and retry lifecycle details.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import types
import uuid
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

logger = logging.getLogger(__name__)


class ProtocolError(ValueError):
    """Invalid client input, before changing policy state."""


class StateConflict(RuntimeError):
    """Stale request or an invalid episode/attempt transition."""


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProtocolError(f"{name} must be a non-negative integer")
    return value


def decode_observation(data):
    """Accept both FastWAM json_numpy strings and ordinary JSON lists."""
    import json_numpy

    arrays = {}
    for key in ("image0", "image1", "image2", "proprio"):
        if key not in data:
            raise ProtocolError(f"Missing field: {key}")
        try:
            value = data[key]
            value = json_numpy.loads(value) if isinstance(value, str) else value
            array = np.asarray(value)
            if array.dtype.kind not in "uif" or not np.isfinite(array).all():
                raise ValueError("expected finite numeric values")
            if key == "proprio":
                if array.shape != (14,):
                    raise ValueError("expected shape [14]")
                array = array.astype(np.float32)
                if not np.isfinite(array).all():
                    raise ValueError("values exceed float32 range")
            else:
                if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) < 1:
                    raise ValueError("expected a nonempty RGB HxWx3 image")
                # RoboTwin sends uint8 RGB. Explicitly reject normalized floats
                # rather than silently applying a second normalization.
                if array.min() < 0 or array.max() > 255 or np.any(array != np.floor(array)):
                    raise ValueError("expected integer RGB values in [0,255]")
                array = array.astype(np.uint8)
            arrays[key] = array
        except (TypeError, ValueError, KeyError, OverflowError) as exc:
            raise ProtocolError(f"Invalid {key}: {exc}") from exc
    return {
        "observation": {
            name: {"rgb": arrays[f"image{i}"]}
            for i, name in enumerate(("head_camera", "left_camera", "right_camera"))
        },
        "joint_action": {"vector": arrays["proprio"]},
    }


class _RemoteEnvironment:
    """Use the existing policy's step() without importing the simulator."""

    def __init__(self, instruction, observation):
        self.instruction = instruction
        self.now_obs = observation
        self.action = None

    def get_instruction(self):
        return self.instruction

    def take_action(self, action, action_type="qpos"):
        if action_type != "qpos" or self.action is not None:
            raise RuntimeError("Expected exactly one qpos action per policy step")
        self.action = np.asarray(action, dtype=np.float32)
        if self.action.shape != (14,) or not np.isfinite(self.action).all():
            raise RuntimeError("Policy returned an invalid action; expected finite [14]")


class ICLWAMServer:
    """One model and one environment stream, serialized across HTTP requests."""

    def __init__(self, policy, vae_device_mode="gpu", pim_trace=None):
        self.policy = policy
        self.vae_device_mode = vae_device_mode
        self.pim_trace = Path(pim_trace).expanduser().resolve() if pim_trace else None
        if self.pim_trace is not None:
            self.pim_trace.parent.mkdir(parents=True, exist_ok=True)
            self.pim_trace.touch(exist_ok=True)
        self.lock = threading.RLock()
        self.session_id = None
        self.attempt_id = 0
        self.next_step = 0
        self.instruction = None
        self.finalized = False
        self.faulted = False
        self.last_reply = None

    def _record_pim_retrieval(self, step):
        if self.pim_trace is None or self.policy.lifecycle is None:
            return
        snapshot = self.policy.lifecycle.last_retrieval
        if snapshot is None:
            return
        matches = [
            {
                "rank": rank,
                "score": float(snapshot["scores"][rank]),
                "source": source,
                "phase": snapshot["phases"][rank].tolist(),
                "effect": snapshot["effects"][rank].tolist(),
            }
            for rank, source in enumerate(snapshot["sources"])
        ]
        event = {
            "session_id": self.session_id,
            "attempt_id": self.attempt_id,
            "step_id": step,
            "instruction": self.instruction,
            "mode": self.policy.zeva_mode,
            "pim_path_enabled": self.policy.zeva_mode == "pim_on",
            "pim_entry_count": len(self.policy.lifecycle.pim),
            "query_phase": snapshot["query_phase"].tolist(),
            "matches": matches,
        }
        with self.pim_trace.open("a", encoding="utf-8") as output:
            output.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def _check_session(self, data):
        if data.get("session_id") is not None and data["session_id"] != self.session_id:
            raise StateConflict("Stale session_id; start or join the current attempt with /reset")
        if self.faulted:
            raise StateConflict("Inference failed; reset the environment and call /reset with attempt_id=0")

    def reset(self, data):
        attempt = _integer(data.get("attempt_id", 0), "attempt_id")
        if attempt >= self.policy.max_attempts:
            raise ProtocolError(f"attempt_id must be below max_attempts={self.policy.max_attempts}")
        if attempt:
            self._check_session(data)
            if (not self.session_id or data.get("session_id") != self.session_id
                    or not self.finalized or attempt != self.attempt_id + 1):
                raise StateConflict("Retry requires the finalized session_id and consecutive attempt_id")
        self.policy.reset()
        self.policy.begin_attempt(attempt)
        self.session_id = uuid.uuid4().hex
        self.attempt_id = attempt
        self.next_step = 0
        self.finalized = False
        self.faulted = False
        self.last_reply = None
        if attempt == 0:
            self.instruction = None
        return {"status": "success", "session_id": self.session_id, "attempt_id": attempt}

    def act(self, data):
        self._check_session(data)
        if self.finalized:
            raise StateConflict("Attempt is finalized; call /reset before /act")
        instruction = data.get("language_instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ProtocolError("language_instruction must be a nonempty string")
        if self.instruction is not None and instruction != self.instruction:
            raise StateConflict("Instruction changed; start a new episode with attempt_id=0")
        step = _integer(data.get("step_id", self.next_step), "step_id")
        if "step_id" in data and not data.get("session_id"):
            raise ProtocolError("step_id requires session_id from /reset")
        if self.last_reply is not None and step == self.next_step - 1:
            return self.last_reply  # Lost HTTP response: do not advance memory twice.
        if step != self.next_step:
            raise StateConflict(f"Expected step_id={self.next_step}, got {step}")
        observation = decode_observation(data)
        if self.session_id is None:
            self.reset({})  # Compatibility with a legacy client's first /act.
        self.instruction = instruction
        env = _RemoteEnvironment(instruction, observation)
        replanning = not self.policy.pending_actions
        try:
            self.policy.step(env, observation)
            if env.action is None:
                raise RuntimeError("Policy did not produce an action")
        except Exception:
            self.faulted = True
            raise
        if replanning:
            try:
                self._record_pim_retrieval(step)
            except (OSError, TypeError, ValueError) as exc:
                logger.warning("Could not write PIM trace to %s: %s", self.pim_trace, exc)
        reply = {
            "action": env.action[None].tolist(),
            "status": "success",
            "session_id": self.session_id,
            "step_id": step,
        }
        self.last_reply = reply
        self.next_step += 1
        return reply

    def finalize(self, data):
        self._check_session(data)
        if not self.session_id or self.instruction is None:
            raise StateConflict("No active attempt to finalize")
        if _integer(data.get("step_id", self.next_step), "step_id") != self.next_step:
            raise StateConflict("Finalize step_id must equal the number of executed actions")
        success = data.get("success", False)
        if not isinstance(success, bool):
            raise ProtocolError("success must be a JSON boolean")
        if not self.finalized:
            observation = decode_observation(data) if any(
                key in data for key in ("image0", "image1", "image2", "proprio")
            ) else None
            env = _RemoteEnvironment(self.instruction, observation)
            try:
                self.policy.finalize_attempt(env, success=success)
            except Exception:
                self.faulted = True
                raise
            self.finalized = True
        return {"status": "success", "session_id": self.session_id, "finalized": True}

    def health_info(self):
        return {
            "status": "error" if self.faulted else "ok",
            "model_loaded": self.policy.model is not None,
            "zeva_mode": self.policy.zeva_mode,
            "vae_device_mode": self.vae_device_mode,
            "action_horizon": self.policy.action_horizon,
            "replan_steps": self.policy.replan_steps,
            "actions_per_response": 1,
            "requires_observation_every_step": True,
            "max_attempts": self.policy.max_attempts,
            "attempt_id": self.attempt_id,
            "next_step_id": self.next_step,
            "finalized": self.finalized,
        }


def create_app(server):
    from flask import Flask, jsonify, request
    from werkzeug.exceptions import BadRequest, UnsupportedMediaType

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

    @app.get("/health")
    def health():
        with server.lock:
            return jsonify(server.health_info())

    def dispatch(method):
        try:
            data = request.get_json()
            if not isinstance(data, dict):
                raise ProtocolError("Request body must be a JSON object")
            with server.lock:
                return jsonify(method(data))
        except (ProtocolError, BadRequest, UnsupportedMediaType) as exc:
            return jsonify(status="error", error=str(exc)), 400
        except StateConflict as exc:
            return jsonify(status="error", error=str(exc)), 409
        except Exception as exc:
            logger.exception("ICL-WAM request failed")
            return jsonify(status="error", error=str(exc)), 500

    app.add_url_rule("/act", "act", lambda: dispatch(server.act), methods=["POST"])
    app.add_url_rule("/reset", "reset", lambda: dispatch(server.reset), methods=["POST"])
    app.add_url_rule("/finalize", "finalize", lambda: dispatch(server.finalize), methods=["POST"])
    return app


def configure_vae(model, mode):
    """Cover BOTH FastWAM image encoding and CTE's direct vae.model.encode."""
    import torch

    for module in model.modules():
        if isinstance(module, torch.nn.Conv3d):
            module._conv_forward = types.MethodType(torch.nn.Conv3d._conv_forward, module)
    if mode == "gpu":
        return
    model.vae.to(device="cpu", dtype=torch.float32).eval()
    original_encode = model.vae.model.encode

    @torch.no_grad()
    def encode_on_cpu(_self, video, scale, *args, **kwargs):
        cpu_scale = [v.to(device="cpu", dtype=torch.float32) if torch.is_tensor(v) else v for v in scale]
        latent = original_encode(video.to(device="cpu", dtype=torch.float32), cpu_scale, *args, **kwargs)
        return latent.to(device=video.device, dtype=video.dtype)

    model.vae.model.encode = types.MethodType(encode_on_cpu, model.vae.model)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "dataset_stats", "cte_checkpoint", "addon_checkpoint"):
        option_names = ["--" + name.replace("_", "-")]
        if "_" in name:
            option_names.append("--" + name)
        parser.add_argument(*option_names, required=name in {"checkpoint", "dataset_stats"})
    parser.add_argument("--task_context_bank", "--task-context-bank")
    parser.add_argument("--task_context_retrieval_checkpoint", "--task-context-retrieval-checkpoint")
    parser.add_argument("--task_context_top_k", "--task-context-top-k", type=int)
    parser.add_argument("--mode", choices=("base", "zeva_stage2", "pim_shadow", "pim_on"), default="pim_on")
    parser.add_argument("--sim_task", "--sim-task", default="robotwin_zeva_fastwam_pim_3cam_384")
    parser.add_argument("--override", action="append", default=[], help="Repeatable Hydra key=value override")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed_precision", "--mixed-precision", choices=("no", "bf16", "fp16"), default="bf16")
    parser.add_argument("--vae_device_mode", "--vae-device-mode", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--model_base_path", "--model-base-path", help="Directory containing Wan-AI pretrained assets")
    parser.add_argument("--allow_download", "--allow-download", action="store_true")
    parser.add_argument("--replan_steps", "--replan-steps", type=int, default=24)
    parser.add_argument("--num_inference_steps", "--num-inference-steps", type=int, default=10)
    parser.add_argument("--max_attempts", "--max-attempts", type=int, default=4)
    parser.add_argument("--pim-trace", help="Append PIM retrievals at each replan to this JSONL file")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def load_policy(args):
    # Configure asset discovery before importing the model/runtime modules.
    if args.model_base_path:
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(Path(args.model_base_path).expanduser().resolve())
    if not args.allow_download:
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for name in ("checkpoint", "dataset_stats", "cte_checkpoint", "addon_checkpoint",
                 "task_context_bank", "task_context_retrieval_checkpoint"):
        value = getattr(args, name)
        if value:
            path = Path(value).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"{name}: {path}")
            setattr(args, name, str(path))
    if args.mode != "base" and not args.cte_checkpoint:
        raise ValueError("--cte_checkpoint is required for non-base modes")
    if args.mode in {"zeva_stage2", "pim_on"} and not args.addon_checkpoint:
        raise ValueError("--addon_checkpoint is required for zeva_stage2/pim_on")
    if args.replan_steps < 1 or args.replan_steps > 32 or (args.mode != "base" and args.replan_steps % 4):
        raise ValueError("replan_steps must be in [1,32], and a multiple of 4 for Zeva")
    if args.num_inference_steps < 1 or args.max_attempts < 1:
        raise ValueError("num_inference_steps and max_attempts must be positive")

    import torch
    from hydra import compose, initialize_config_dir
    from experiments.robotwin.iclwam_policy.robotwin_policy import (
        WorldActionRobotWinPolicy, _mixed_precision_to_model_dtype,
    )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu --mixed_precision no for debugging")
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "configs")):
        cfg = compose(config_name="sim_robotwin_zeva", overrides=[f"task={args.sim_task}", *args.override])
    task_context = cfg.model.zeva.task_context
    for name, key in (("task_context_bank", "bank_path"),
                      ("task_context_retrieval_checkpoint", "retrieval_checkpoint"),
                      ("task_context_top_k", "top_k")):
        if getattr(args, name) is not None:
            task_context[key] = getattr(args, name)
    if args.mode in {"zeva_stage2", "pim_on"} and task_context.mode == "static":
        if not task_context.bank_path or not task_context.retrieval_checkpoint:
            raise ValueError("static task context requires --task_context_bank and --task_context_retrieval_checkpoint")
    dtype = _mixed_precision_to_model_dtype(args.mixed_precision if args.device != "cpu" else "no")
    evaluation = cfg.EVALUATION
    policy = WorldActionRobotWinPolicy(
        model_cfg=cfg.model, processor_cfg=cfg.data.train.processor,
        checkpoint_path=args.checkpoint, dataset_stats_path=Path(args.dataset_stats),
        device=args.device, model_dtype=dtype, action_horizon=32,
        replan_steps=args.replan_steps, num_inference_steps=args.num_inference_steps,
        sigma_shift=evaluation.sigma_shift, seed=args.seed,
        text_cfg_scale=evaluation.text_cfg_scale, negative_prompt=evaluation.negative_prompt,
        rand_device=evaluation.rand_device, tiled=evaluation.tiled, timing_enabled=False,
        num_video_frames=(cfg.data.train.num_frames - 1) // cfg.data.train.action_video_freq_ratio + 1,
        video_size=cfg.data.train.video_size, zeva_mode=args.mode,
        cte_checkpoint=args.cte_checkpoint, addon_checkpoint=args.addon_checkpoint,
        pim_top_k=evaluation.pim_top_k, max_attempts=args.max_attempts,
    )
    configure_vae(policy.model, args.vae_device_mode)
    return policy


def main():
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    policy = load_policy(args)
    server = ICLWAMServer(policy, args.vae_device_mode, pim_trace=args.pim_trace)
    logger.info("ICL-WAM ready: %s", server.health_info())
    create_app(server).run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
