"""Fixed-seed repeated-attempt Zeva evaluation with explicit static-context artifacts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cte-checkpoint")
    parser.add_argument("--addon-checkpoint")
    parser.add_argument("--task-context-bank")
    parser.add_argument("--task-context-retrieval-checkpoint")
    parser.add_argument("--task-context-top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--mode",
        choices=("base", "zeva_stage2", "pim_shadow", "pim_on"),
        default="pim_on",
    )
    parser.add_argument("--max-attempts", type=int, default=4)
    args = parser.parse_args()

    if args.mode != "base" and not args.cte_checkpoint:
        parser.error("--cte-checkpoint is required for non-base Zeva modes")
    if args.mode in {"zeva_stage2", "pim_on"} and not args.addon_checkpoint:
        parser.error("--addon-checkpoint is required for zeva_stage2 and pim_on")
    if args.mode in {"zeva_stage2", "pim_on"}:
        if not args.task_context_bank or not args.task_context_retrieval_checkpoint:
            parser.error(
                "static Zeva evaluation requires --task-context-bank and "
                "--task-context-retrieval-checkpoint"
            )
        if args.task_context_top_k < 1:
            parser.error("--task-context-top-k must be positive")

    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(root / "experiments/robotwin/eval_robotwin_single.py"),
        "--config-name",
        "sim_robotwin_zeva.yaml",
        f"ckpt={args.ckpt}",
        f"EVALUATION.task_name={args.task}",
        f"EVALUATION.fixed_seed={args.seed}",
        f"EVALUATION.max_attempts={args.max_attempts}",
        f"EVALUATION.zeva_mode={args.mode}",
        f"EVALUATION.cte_checkpoint={args.cte_checkpoint}",
    ]
    if args.mode in {"zeva_stage2", "pim_on"}:
        cmd.extend(
            [
                f"EVALUATION.addon_checkpoint={args.addon_checkpoint}",
                f"EVALUATION.task_context_bank={args.task_context_bank}",
                "EVALUATION.task_context_mode=static",
                f"EVALUATION.task_context_retrieval_checkpoint={args.task_context_retrieval_checkpoint}",
                f"EVALUATION.task_context_top_k={args.task_context_top_k}",
            ]
        )
    raise SystemExit(subprocess.call(cmd, cwd=root))


if __name__ == "__main__":
    main()
