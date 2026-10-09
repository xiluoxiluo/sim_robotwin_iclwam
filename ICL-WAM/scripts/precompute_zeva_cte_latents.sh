#!/usr/bin/env bash

set -euo pipefail

cd /data/share/1919650160032350208/zjj/ICL-WAM

export PYTHONPATH="/data/share/1919650160032350208/zjj/ICL-WAM/src:${PYTHONPATH:-}"

export DIFFSYNTH_MODEL_BASE_PATH="/data/share/1919650160032350208/zjj/fastwam/checkpoints"

# ------------------------------------------------------------
# GPU configuration
# ------------------------------------------------------------

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

# Count visible GPUs.
IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
NPROC_PER_NODE="${#GPU_ARRAY[@]}"

echo "=========================================="
echo "VAE latent cache precompute"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "nproc_per_node=${NPROC_PER_NODE}"
echo "=========================================="

# Avoid CPU oversubscription when several PyAV workers are used.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/precompute_zeva_cte_latents.py \
  task=robotwin_zeva_fastwam_3cam_384 \
  +model.zeva.cte.latent_cache_path=/data/share/1919650160032350208/foundation_model/datasets/robotwin2.0/zeva_cte_latent_cache_v1 \
  +cte_latent_cache.batch_size=8 \
  +cte_latent_cache.num_workers=8 \
  +cte_latent_cache.prefetch_factor=2 \
  +cte_latent_cache.flush_every_batches=8 \
  +cte_latent_cache.presample_images=true \
  +cte_latent_cache.overwrite=false