"""Fixed-seed repeated-attempt evaluation entry point."""

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
    parser.add_argument("--task-context-top-k", type=int)
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
    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(root / "experiments/robotwin/eval_robotwin_single.py"),
           "--config-name", "sim_robotwin_zeva.yaml", f"ckpt={args.ckpt}",
           f"EVALUATION.task_name={args.task}", f"EVALUATION.fixed_seed={args.seed}",
           f"EVALUATION.max_attempts={args.max_attempts}", f"EVALUATION.zeva_mode={args.mode}",
           f"EVALUATION.cte_checkpoint={args.cte_checkpoint}"]
    if args.mode in {"zeva_stage2", "pim_on"}:
        cmd.append(f"EVALUATION.addon_checkpoint={args.addon_checkpoint}")
    if args.task_context_bank:
        cmd.extend([
            "model.zeva.task_context.mode=bank",
            f"model.zeva.task_context.bank_path={args.task_context_bank}",
        ])
    if args.task_context_retrieval_checkpoint:
        if not args.task_context_bank:
            parser.error("--task-context-retrieval-checkpoint requires --task-context-bank")
        cmd.extend([
            "model.zeva.task_context.mode=static",
            f"model.zeva.task_context.retrieval_checkpoint={args.task_context_retrieval_checkpoint}",
        ])
    task_context_top_k = args.task_context_top_k
    if task_context_top_k is None and args.task_context_retrieval_checkpoint:
        task_context_top_k = 5
    if task_context_top_k is not None:
        if task_context_top_k < 1:
            parser.error("--task-context-top-k must be positive")
        cmd.append(f"model.zeva.task_context.top_k={task_context_top_k}")
    raise SystemExit(subprocess.call(cmd, cwd=root))


if __name__ == "__main__":
    main()
