#!/usr/bin/env python3
"""Retry one RoboTwin simulator seed until success, preserving ICL-WAM PIM."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlsplit
from pathlib import Path


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def health(url):
    with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=30) as response:
        return json.load(response)


def attempt_result(path, task, sim_seed):
    """Read the one stable rollout expected from an isolated attempt directory."""
    with path.open() as result_file:
        results = [json.loads(line) for line in result_file if line.strip()]
    if len(results) != 1 or results[0].get("seed") != sim_seed:
        raise ValueError(f"Expected exactly one result for seed={sim_seed} in {path}")
    status = results[0].get(f"sim/{task}")
    if type(status) is not int or status not in (0, 1):
        raise ValueError(f"Expected a binary task result in {path}, got {status!r}")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="put_object_cabinet")
    parser.add_argument("--task-config", default="demo_randomized")
    parser.add_argument("--sim-seed", type=int, default=2002)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--server-url", default="http://127.0.0.1:8765")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=360000000)
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument("--output-dir", default="/home/ubuntu/robotwin_realted/log_iclwam_retry")
    parser.add_argument("--client-dir", default="/home/ubuntu/robotwin_realted/client")
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error("--attempts must be positive")
    if args.sim_seed < 0:
        parser.error("--sim-seed must be nonnegative")

    root = Path(args.output_dir).expanduser().resolve()
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{args.task}_seed_{args.sim_seed}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_at": now(), "run_id": run_id, "task": args.task,
        "task_config": args.task_config, "sim_seed": args.sim_seed,
        "attempts_requested": args.attempts,
        "server_url": args.server_url, "client_script": "client_skip_unstable_seed_iclwam.py",
        "command": sys.argv, "attempts": [],
    }
    (run_dir / "config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        manifest["initial_health"] = health(args.server_url)
    except Exception as exc:
        manifest["initial_health_error"] = repr(exc)
        (run_dir / "config.json").write_text(json.dumps(manifest, indent=2) + "\n")
        raise SystemExit(f"ICL-WAM server is not healthy: {exc}")
    server_max_attempts = manifest["initial_health"].get("max_attempts")
    if isinstance(server_max_attempts, int) and args.attempts > server_max_attempts:
        message = (f"--attempts={args.attempts} exceeds the server's max_attempts="
                   f"{server_max_attempts}; restart the server with --max_attempts {args.attempts}")
        manifest["error"] = message
        (run_dir / "config.json").write_text(json.dumps(manifest, indent=2) + "\n")
        raise SystemExit(message)

    print(f"run_dir={run_dir}")
    print(f"sim_seed={args.sim_seed}")
    client = Path(args.client_dir).expanduser().resolve() / "client_skip_unstable_seed_iclwam.py"
    server_parts = urlsplit(args.server_url)
    if server_parts.hostname is None or server_parts.port is None:
        raise SystemExit("--server-url must include host and port, e.g. http://127.0.0.1:8765")
    previous_session_id = None
    fixed_instruction = None
    for attempt in range(args.attempts):
        started_at = now()
        started = time.time()
        attempt_dir = run_dir / f"attempt_{attempt:02d}"
        attempt_dir.mkdir()
        command = [sys.executable, str(client), "--host", server_parts.hostname,
                   "--port", str(server_parts.port),
                   "--eval_log_dir", str(attempt_dir), "--num_episodes", "1",
                   "--device", str(args.device), "--seed", "0",
                   "--fixed_sim_seed", str(args.sim_seed),
                   "--task_name", args.task, "--output_path", str(attempt_dir),
                   "--task_config", args.task_config, "--max_seed_trials", "1",
                   "--iclwam_http", "--iclwam_attempt_id", str(attempt),
                   "--iclwam_session_file", str(attempt_dir / "session_id.txt")]
        if previous_session_id is not None:
            command.extend(["--iclwam_session_id", previous_session_id,
                            "--fixed_instruction", fixed_instruction])
        log_path = attempt_dir / "client.log"
        print(f"attempt={attempt} starting; log={log_path}", flush=True)
        with log_path.open("w") as log:
            proc = subprocess.run(command, cwd=args.client_dir, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=args.timeout)
        entry = {"attempt": attempt, "started_at": started_at, "returncode": proc.returncode,
                 "duration_s": round(time.time() - started, 3), "command": command,
                 "log": str(log_path)}
        try:
            entry["health_after"] = health(args.server_url)
        except Exception as exc:
            entry["health_after_error"] = repr(exc)
        try:
            if proc.returncode != 0:
                raise RuntimeError(f"Client exited with code {proc.returncode}")
            seed_record = json.loads((attempt_dir / args.task / "seed_record.json").read_text())
            skipped = seed_record.get("skipped_seeds", [])
            if skipped:
                entry["skipped_seeds"] = skipped
                reasons = ", ".join(str(item.get("reason_type", "unknown")) for item in skipped)
                raise ValueError(f"Seed {args.sim_seed} was skipped before policy rollout: {reasons}; "
                                 "no PIM retry is possible for this run")
            entry["task_status"] = attempt_result(attempt_dir / "results.json", args.task, args.sim_seed)
            episodes = seed_record.get("counted_episodes", [])
            if len(episodes) != 1 or episodes[0].get("seed") != args.sim_seed:
                raise ValueError("Expected one counted episode for the requested seed")
            instruction = episodes[0].get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("Missing instruction for the fixed-seed episode")
            if fixed_instruction is not None and instruction != fixed_instruction:
                raise ValueError("Retry used a different instruction")
            fixed_instruction = instruction
            session_id = (attempt_dir / "session_id.txt").read_text().strip()
            if not session_id:
                raise ValueError("Client did not record a server session ID")
            entry["session_id"] = session_id
            if entry["health_after"].get("attempt_id") != attempt or not entry["health_after"].get("finalized"):
                raise ValueError("Server did not finalize the expected attempt")
            entry["success"] = entry["task_status"] == 1
            previous_session_id = session_id
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            entry["error"] = str(exc)
            manifest["error"] = f"attempt {attempt}: {exc}"
        manifest["attempts"].append(entry)
        (run_dir / "config.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"attempt={attempt} returncode={proc.returncode} task_status={entry.get('task_status')}"
              f" error={entry.get('error')}", flush=True)
        if "error" in entry:
            break
        if entry["success"]:
            manifest["successful_attempt"] = attempt
            break
        if attempt + 1 < args.attempts:
            time.sleep(args.sleep)
    manifest["finished_at"] = now()
    (run_dir / "config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"run_dir": str(run_dir), "attempts": len(manifest["attempts"]),
                      "successful_attempt": manifest.get("successful_attempt")}, indent=2))
    return 0 if "successful_attempt" in manifest else 1


if __name__ == "__main__":
    raise SystemExit(main())
