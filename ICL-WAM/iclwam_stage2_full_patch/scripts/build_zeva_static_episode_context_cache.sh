#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
: "${BEHAVIOR_BANK:?Set BEHAVIOR_BANK}"
: "${READOUT_CACHE:?Set READOUT_CACHE}"
: "${RETRIEVAL_HEAD:?Set RETRIEVAL_HEAD}"
: "${STATIC_CONTEXT_CACHE:?Set STATIC_CONTEXT_CACHE}"

exec python scripts/build_zeva_static_episode_context_cache.py \
  task=robotwin_zeva_fastwam_stage2_3cam_384 \
  ++model.zeva.task_context.bank_path="$BEHAVIOR_BANK" \
  ++model.zeva.task_context.readout_cache_path="$READOUT_CACHE" \
  ++model.zeva.task_context.retrieval_checkpoint="$RETRIEVAL_HEAD" \
  ++model.zeva.task_context.static_episode_cache_path="$STATIC_CONTEXT_CACHE"
