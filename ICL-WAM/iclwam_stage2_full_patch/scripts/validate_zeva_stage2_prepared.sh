#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/data/share/1919650160032350208/zjj/fastwam/checkpoints}"

: "${BASE_CKPT:?Set BASE_CKPT}"
: "${CTE_CKPT:?Set CTE_CKPT}"
: "${PHASE_CACHE:?Set PHASE_CACHE}"
: "${BEHAVIOR_BANK:?Set BEHAVIOR_BANK}"
: "${READOUT_CACHE:?Set READOUT_CACHE}"
: "${RETRIEVAL_HEAD:?Set RETRIEVAL_HEAD}"
: "${STATIC_CONTEXT_CACHE:?Set STATIC_CONTEXT_CACHE}"
: "${PIM_CACHE:?Set PIM_CACHE}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec python scripts/validate_zeva_stage2_prepared.py \
  task=robotwin_zeva_fastwam_stage2_3cam_384 \
  ckpt="$BASE_CKPT" \
  ++model.zeva.cte.checkpoint="$CTE_CKPT" \
  ++model.zeva.cache.path="$PHASE_CACHE" \
  ++model.zeva.task_context.bank_path="$BEHAVIOR_BANK" \
  ++model.zeva.task_context.readout_cache_path="$READOUT_CACHE" \
  ++model.zeva.task_context.retrieval_checkpoint="$RETRIEVAL_HEAD" \
  ++model.zeva.task_context.static_episode_cache_path="$STATIC_CONTEXT_CACHE" \
  ++model.zeva.prepared_stage2.pim_cache_path="$PIM_CACHE" \
  "$@"
