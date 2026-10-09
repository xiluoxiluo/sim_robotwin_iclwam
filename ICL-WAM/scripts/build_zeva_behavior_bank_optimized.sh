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

: "${BASE_CKPT:?Set BASE_CKPT to the frozen FastWAM checkpoint}"
: "${CTE_CKPT:?Set CTE_CKPT to the frozen Stage-1 CTE checkpoint}"
: "${LATENT_CACHE:?Set LATENT_CACHE to zeva_cte_latent_cache_v1}"
: "${BEHAVIOR_BANK:?Set BEHAVIOR_BANK output .pt path}"
: "${READOUT_CACHE:?Set READOUT_CACHE output .pt path}"

exec torchrun --standalone --nproc_per_node="$WORLD_SIZE" \
  scripts/build_zeva_behavior_bank_optimized.py \
  task=robotwin_zeva_fastwam_stage2_3cam_384 \
  ++ckpt="$BASE_CKPT" \
  ++model.zeva.cte.checkpoint="$CTE_CKPT" \
  ++model.zeva.cte.latent_cache_path="$LATENT_CACHE" \
  ++model.zeva.task_context.bank_path="$BEHAVIOR_BANK" \
  ++model.zeva.task_context.readout_cache_path="$READOUT_CACHE"
