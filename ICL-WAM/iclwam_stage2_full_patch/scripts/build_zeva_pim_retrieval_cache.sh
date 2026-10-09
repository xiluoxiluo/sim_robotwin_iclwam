#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
WORLD_SIZE="${#GPU_ARRAY[@]}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

: "${PHASE_CACHE:?Set PHASE_CACHE to zeva_phase_effect_cache_v4}"
: "${CTE_CKPT:?Set CTE_CKPT to the frozen Stage-1 CTE checkpoint}"
: "${PIM_CACHE:?Set PIM_CACHE output directory}"

exec torchrun --standalone --nproc_per_node="$WORLD_SIZE" \
  scripts/build_zeva_pim_retrieval_cache.py \
  task=robotwin_zeva_fastwam_stage2_3cam_384 \
  ++model.zeva.cache.path="$PHASE_CACHE" \
  ++model.zeva.cte.checkpoint="$CTE_CKPT" \
  ++model.zeva.prepared_stage2.pim_cache_path="$PIM_CACHE"
