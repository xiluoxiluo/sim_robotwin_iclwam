#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/data/share/1919650160032350208/zjj/fastwam/checkpoints}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
WORLD_SIZE="${#GPU_ARRAY[@]}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

# Resolve one run directory in the parent shell so all torchrun ranks share
# exactly the same checkpoint/metrics/state destination.
RUN_DIR="${RUN_DIR:-$ROOT/runs/robotwin_zeva_fastwam_stage2_3cam_384/$(date +%Y-%m-%d_%H-%M-%S)}"
mkdir -p "$RUN_DIR"

: "${BASE_CKPT:?Set BASE_CKPT}"
: "${CTE_CKPT:?Set CTE_CKPT}"
: "${PHASE_CACHE:?Set PHASE_CACHE to the validated v4 cache}"
: "${BEHAVIOR_BANK:?Set BEHAVIOR_BANK}"
: "${READOUT_CACHE:?Set READOUT_CACHE}"
: "${RETRIEVAL_HEAD:?Set RETRIEVAL_HEAD}"
: "${STATIC_CONTEXT_CACHE:?Set STATIC_CONTEXT_CACHE}"
: "${PIM_CACHE:?Set PIM_CACHE}"

exec torchrun --standalone --nproc_per_node="$WORLD_SIZE" \
  scripts/train_zeva_fastwam_prepared.py \
  task=robotwin_zeva_fastwam_stage2_3cam_384 \
  output_dir="$RUN_DIR" \
  ckpt="$BASE_CKPT" \
  ++model.zeva.cte.checkpoint="$CTE_CKPT" \
  ++model.zeva.cache.path="$PHASE_CACHE" \
  ++model.zeva.task_context.bank_path="$BEHAVIOR_BANK" \
  ++model.zeva.task_context.readout_cache_path="$READOUT_CACHE" \
  ++model.zeva.task_context.retrieval_checkpoint="$RETRIEVAL_HEAD" \
  ++model.zeva.task_context.static_episode_cache_path="$STATIC_CONTEXT_CACHE" \
  ++model.zeva.prepared_stage2.pim_cache_path="$PIM_CACHE" \
  "$@"
