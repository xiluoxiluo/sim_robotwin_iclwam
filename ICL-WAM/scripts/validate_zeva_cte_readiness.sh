#!/usr/bin/env bash
set -euo pipefail

cd /data/share/1919650160032350208/zjj/ICL-WAM
export PYTHONPATH=/data/share/1919650160032350208/zjj/ICL-WAM/src:${PYTHONPATH:-}
export DIFFSYNTH_MODEL_BASE_PATH=/data/share/1919650160032350208/zjj/fastwam/checkpoints

CTE_CKPT="${CTE_CKPT:-runs/robotwin_zeva_fastwam_3cam_384/2026-09-23_20-32-07/cte.pt}"
LATENT_CACHE="${LATENT_CACHE:-/data/share/1919650160032350208/foundation_model/datasets/robotwin2.0/zeva_cte_latent_cache_v1}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python scripts/validate_zeva_cte_readiness.py \
  task=robotwin_zeva_fastwam_3cam_384 \
  ++model.zeva.cte.checkpoint="${CTE_CKPT}" \
  ++model.zeva.cte.latent_cache_path="${LATENT_CACHE}" \
  ++cte_readiness.sample_windows=2048 \
  ++cte_readiness.batch_size=64 \
  ++cte_readiness.full_prefix_episodes=32 \
  ++cte_readiness.windows_per_episode=4 \
  ++cte_readiness.strict_local_full=false
