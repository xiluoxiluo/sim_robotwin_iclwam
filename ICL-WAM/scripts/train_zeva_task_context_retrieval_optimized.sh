#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
: "${BEHAVIOR_BANK:?Set BEHAVIOR_BANK}"
: "${READOUT_CACHE:?Set READOUT_CACHE}"
: "${RETRIEVAL_HEAD:?Set RETRIEVAL_HEAD}"

exec python scripts/train_zeva_task_context_retrieval_optimized.py \
  task=robotwin_zeva_fastwam_stage2_3cam_384 \
  ++model.zeva.task_context.bank_path="$BEHAVIOR_BANK" \
  ++model.zeva.task_context.readout_cache_path="$READOUT_CACHE" \
  ++model.zeva.task_context.retrieval_checkpoint="$RETRIEVAL_HEAD"
