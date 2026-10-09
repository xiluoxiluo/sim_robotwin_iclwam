"""Apply small evaluation/config patches to the current ICL-WAM checkout.

The large deploy/eval files are edited by exact anchors so the patch refuses to
apply to an unexpected repository revision instead of silently corrupting code.
The script is idempotent.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str, marker: str) -> None:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    if old not in text:
        raise RuntimeError(f"patch anchor not found in {path}: {old[:120]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    # ------------------------------------------------------------------
    # sim_robotwin_zeva.yaml: declare explicit delegated evaluation fields.
    # ------------------------------------------------------------------
    path = ROOT / "configs/sim_robotwin_zeva.yaml"
    replace_once(
        path,
        "  addon_checkpoint: null\n",
        "  addon_checkpoint: null\n"
        "  # Explicitly forwarded to the delegated RoboTwin policy process.\n"
        "  task_context_mode: static\n"
        "  task_context_bank: null\n"
        "  task_context_retrieval_checkpoint: null\n"
        "  task_context_top_k: 5\n",
        "task_context_retrieval_checkpoint:",
    )

    # ------------------------------------------------------------------
    # deploy_policy.yml: expose the same three artifacts to RoboTwin CLI.
    # ------------------------------------------------------------------
    path = ROOT / "experiments/robotwin/fastwam_policy/deploy_policy.yml"
    replace_once(
        path,
        "addon_checkpoint: null\n",
        "addon_checkpoint: null\n"
        "task_context_mode: static\n"
        "task_context_bank: null\n"
        "task_context_retrieval_checkpoint: null\n"
        "task_context_top_k: 5\n",
        "task_context_retrieval_checkpoint:",
    )

    # ------------------------------------------------------------------
    # eval_robotwin_single.py: forward task-context artifacts to subprocess.
    # ------------------------------------------------------------------
    path = ROOT / "experiments/robotwin/eval_robotwin_single.py"
    old = 'for key in ("zeva_mode", "cte_checkpoint", "addon_checkpoint", "pim_top_k", "max_attempts"):'
    new = (
        'for key in ("zeva_mode", "cte_checkpoint", "addon_checkpoint", "pim_top_k", "max_attempts", '
        '"task_context_mode", "task_context_bank", "task_context_retrieval_checkpoint", "task_context_top_k"):'
    )
    replace_once(path, old, new, "task_context_retrieval_checkpoint\", \"task_context_top_k")
    replace_once(
        path,
        '            if key in {"cte_checkpoint", "addon_checkpoint"}:\n',
        '            if key in {"cte_checkpoint", "addon_checkpoint", "task_context_bank", "task_context_retrieval_checkpoint"}:\n',
        '"task_context_bank", "task_context_retrieval_checkpoint"}:',
    )

    # ------------------------------------------------------------------
    # deploy_policy.py: accept explicit task-context overrides and place them
    # into the composed model config before FastWAM is instantiated.
    # ------------------------------------------------------------------
    path = ROOT / "experiments/robotwin/fastwam_policy/deploy_policy.py"
    replace_once(
        path,
        "        pim_top_k: int = 4,\n        max_attempts: int = 1,\n    ) -> None:\n"
        "        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))\n"
        "        model_cfg_copy.load_text_encoder = True\n",
        "        pim_top_k: int = 4,\n        max_attempts: int = 1,\n"
        "        task_context_mode: Optional[str] = None,\n"
        "        task_context_bank: Optional[str] = None,\n"
        "        task_context_retrieval_checkpoint: Optional[str] = None,\n"
        "        task_context_top_k: Optional[int] = None,\n"
        "    ) -> None:\n"
        "        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))\n"
        "        model_cfg_copy.load_text_encoder = True\n"
        "        # Delegated RoboTwin evaluation runs in a separate process.\n"
        "        # Apply explicit static-context artifacts before model/session setup\n"
        "        # so evaluation uses the same bank/head/top-k identity as Stage-2.\n"
        "        if not _is_none_like(task_context_mode):\n"
        "            model_cfg_copy.zeva.task_context.mode = str(task_context_mode)\n"
        "        if not _is_none_like(task_context_bank):\n"
        "            model_cfg_copy.zeva.task_context.bank_path = str(task_context_bank)\n"
        "        if not _is_none_like(task_context_retrieval_checkpoint):\n"
        "            model_cfg_copy.zeva.task_context.retrieval_checkpoint = str(task_context_retrieval_checkpoint)\n"
        "        if task_context_top_k is not None:\n"
        "            model_cfg_copy.zeva.task_context.top_k = int(task_context_top_k)\n",
        "Delegated RoboTwin evaluation runs in a separate process.",
    )

    replace_once(
        path,
        '    max_attempts = int(usr_args.get("max_attempts", cfg.EVALUATION.get("max_attempts", 1)))\n',
        '    max_attempts = int(usr_args.get("max_attempts", cfg.EVALUATION.get("max_attempts", 1)))\n'
        '    task_context_mode = usr_args.get("task_context_mode", cfg.EVALUATION.get("task_context_mode"))\n'
        '    task_context_bank = usr_args.get("task_context_bank", cfg.EVALUATION.get("task_context_bank"))\n'
        '    task_context_retrieval_checkpoint = usr_args.get(\n'
        '        "task_context_retrieval_checkpoint", cfg.EVALUATION.get("task_context_retrieval_checkpoint")\n'
        '    )\n'
        '    task_context_top_k = _parse_optional_int(\n'
        '        usr_args.get("task_context_top_k", cfg.EVALUATION.get("task_context_top_k"))\n'
        '    )\n',
        'task_context_retrieval_checkpoint = usr_args.get(',
    )

    replace_once(
        path,
        "        pim_top_k=pim_top_k,\n        max_attempts=max_attempts,\n    )\n",
        "        pim_top_k=pim_top_k,\n        max_attempts=max_attempts,\n"
        "        task_context_mode=None if _is_none_like(task_context_mode) else str(task_context_mode),\n"
        "        task_context_bank=None if _is_none_like(task_context_bank) else str(task_context_bank),\n"
        "        task_context_retrieval_checkpoint=(\n"
        "            None if _is_none_like(task_context_retrieval_checkpoint)\n"
        "            else str(task_context_retrieval_checkpoint)\n"
        "        ),\n"
        "        task_context_top_k=task_context_top_k,\n"
        "    )\n",
        "task_context_top_k=task_context_top_k,",
    )

    print("Stage-2 evaluation/config patch: PASSED")


if __name__ == "__main__":
    main()
