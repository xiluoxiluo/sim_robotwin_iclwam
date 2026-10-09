"""RoboTwin HTTP client for iclwam_server.py (no torch/model dependency)."""
from __future__ import annotations

import time

import json_numpy
import numpy as np
import requests


def _observation_payload(observation):
    cameras = observation["observation"]
    return {
        **{f"image{i}": json_numpy.dumps(np.asarray(cameras[name]["rgb"]))
           for i, name in enumerate(("head_camera", "left_camera", "right_camera"))},
        "proprio": json_numpy.dumps(np.asarray(observation["joint_action"]["vector"], dtype=np.float32)),
    }


class ICLWAMHTTPPolicy:
    def __init__(self, server_url="http://127.0.0.1:8765", timeout=300.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = float(timeout)
        if self.timeout <= 0:
            raise ValueError("http_timeout must be positive")
        self.http = requests.Session()
        # Local simulator traffic must not be routed through Clash/system proxies.
        self.http.trust_env = False
        self.session_id = None
        self.step_id = 0
        self._needs_reset = True
        self._finalized = False
        self._timing = {"infer_s": 0.0, "sim_s": 0.0}
        response = self.http.get(self.server_url + "/health", timeout=self.timeout)
        response.raise_for_status()
        health = response.json()
        if not health.get("model_loaded") or health.get("actions_per_response") != 1:
            raise RuntimeError(f"Expected an ICL-WAM single-action server: {health}")

    def _post(self, endpoint, payload):
        response = self.http.post(self.server_url + endpoint, json=payload, timeout=self.timeout)
        if not response.ok:
            raise RuntimeError(f"ICL-WAM {endpoint}: HTTP {response.status_code}: {response.text}")
        result = response.json()
        if result.get("status") != "success":
            raise RuntimeError(f"ICL-WAM {endpoint}: {result}")
        return result

    def reset(self):
        # RoboTwin calls reset_model() immediately before begin_attempt().
        # Defer /reset so attempt_id>0 retains the previous attempt's PIM.
        self._needs_reset = True
        self.step_id = 0
        self.reset_timing_rollout()

    def begin_attempt(self, attempt_id):
        payload = {"attempt_id": int(attempt_id)}
        if attempt_id:
            payload["session_id"] = self.session_id
        result = self._post("/reset", payload)
        self.session_id = result["session_id"]
        self.step_id = 0
        self._needs_reset = False
        self._finalized = False

    def should_request_observation(self):
        return True

    def step(self, task_env, observation):
        if observation is None:
            raise ValueError("ICL-WAM requires a fresh observation at every control step")
        if self._needs_reset:
            self.begin_attempt(0)
        if self._finalized:
            raise RuntimeError("Attempt is finalized; reset before stepping")
        payload = _observation_payload(observation)
        payload.update(language_instruction=task_env.get_instruction(),
                       session_id=self.session_id, step_id=self.step_id)
        start = time.perf_counter()
        # A lost response can safely retry the SAME step_id: the server caches
        # its last action and never advances CTE/BIT/PIM for a duplicate.
        try:
            result = self._post("/act", payload)
        except (requests.Timeout, requests.ConnectionError):
            result = self._post("/act", payload)
        action = np.asarray(result["action"], dtype=np.float32)
        if (action.shape != (1, 14) or not np.isfinite(action).all()
                or result.get("session_id") != self.session_id
                or result.get("step_id") != self.step_id):
            raise RuntimeError("Invalid ICL-WAM action shape or session/step identity")
        self._timing["infer_s"] += time.perf_counter() - start
        start = time.perf_counter()
        task_env.take_action(action[0], action_type="qpos")
        self._timing["sim_s"] += time.perf_counter() - start
        self.step_id += 1

    def finalize_attempt(self, task_env, success=False):
        if self._needs_reset or self._finalized:
            return
        payload = {"session_id": self.session_id, "step_id": self.step_id, "success": bool(success)}
        # The evaluator exits right after take_action(); now_obs is its actual
        # terminal after-frame. Never substitute a stale pre-action frame.
        terminal_obs = getattr(task_env, "now_obs", None)
        if isinstance(terminal_obs, dict) and "observation" in terminal_obs:
            payload.update(_observation_payload(terminal_obs))
        self._post("/finalize", payload)
        self._finalized = True

    def reset_timing_rollout(self):
        self._timing = {"infer_s": 0.0, "sim_s": 0.0}

    def get_timing_rollout(self):
        return dict(self._timing)


def get_model(usr_args):
    return ICLWAMHTTPPolicy(
        server_url=usr_args.get("server_url", "http://127.0.0.1:8765"),
        timeout=usr_args.get("http_timeout", 300),
    )


def eval(TASK_ENV, model, observation):
    model.step(TASK_ENV, observation)


def reset_model(model):
    model.reset()
