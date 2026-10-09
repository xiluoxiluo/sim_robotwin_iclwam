#!/usr/bin/env bash
set -euo pipefail

cd /data/share/1919650160032350208/zjj/ICL-WAM
export PYTHONPATH=/data/share/1919650160032350208/zjj/ICL-WAM/src:${PYTHONPATH:-}
export DIFFSYNTH_MODEL_BASE_PATH=/data/share/1919650160032350208/zjj/fastwam/checkpoints

# Examples:
#   GPU_IDS=0,1 bash scripts/build_zeva_robotwin_cache_from_latents.sh
#   GPU_IDS=0,1,2,...,15 bash scripts/build_zeva_robotwin_cache_from_latents.sh
GPU_IDS="${GPU_IDS:-0,1}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
NPROC_PER_NODE="${#GPU_ARRAY[@]}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-1}"

CTE_CKPT="${CTE_CKPT:-runs/robotwin_zeva_fastwam_3cam_384/2026-09-23_20-32-07/cte.pt}"
LATENT_CACHE="${LATENT_CACHE:-/data/share/1919650160032350208/foundation_model/datasets/robotwin2.0/zeva_cte_latent_cache_v1}"
PHASE_CACHE="${PHASE_CACHE:-/data/share/1919650160032350208/foundation_model/datasets/robotwin2.0/zeva_phase_effect_cache_v4}"
TAIL_BATCH_SIZE="${TAIL_BATCH_SIZE:-4}"
TAIL_NUM_WORKERS="${TAIL_NUM_WORKERS:-2}"

cat <<EOF
============================================================
Zeva phase/effect cache-v4 from VAE latent cache
GPU_IDS        : ${GPU_IDS}
world size     : ${NPROC_PER_NODE}
CTE checkpoint : ${CTE_CKPT}
latent cache   : ${LATENT_CACHE}
output cache   : ${PHASE_CACHE}
tail batch     : ${TAIL_BATCH_SIZE}
tail workers   : ${TAIL_NUM_WORKERS}/rank
============================================================
EOF

exec python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/build_zeva_robotwin_cache_from_latents.py \
  task=robotwin_zeva_fastwam_3cam_384 \
  ++model.zeva.cte.checkpoint="${CTE_CKPT}" \
  ++model.zeva.cte.latent_cache_path="${LATENT_CACHE}" \
  ++model.zeva.cache.path="${PHASE_CACHE}" \
  ++model.zeva.cache.overwrite=false \
  ++phase_cache_from_latents.tail_batch_size="${TAIL_BATCH_SIZE}" \
  ++phase_cache_from_latents.tail_num_workers="${TAIL_NUM_WORKERS}" \
  ++phase_cache_from_latents.tail_prefetch_factor=2
