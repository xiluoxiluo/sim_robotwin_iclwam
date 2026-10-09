#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Zeva RoboTwin phase/effect cache builder
# Machine target:
#   128 CPU cores
#   16 PPU
# ============================================================

cd /data/share/1919650160032350208/zjj/ICL-WAM

# ------------------------------------------------------------
# FastWAM / Python environment
# ------------------------------------------------------------
export DIFFSYNTH_MODEL_BASE_PATH=/data/share/1919650160032350208/zjj/fastwam/checkpoints
export PYTHONPATH=/data/share/1919650160032350208/zjj/ICL-WAM/src:${PYTHONPATH:-}

# ------------------------------------------------------------
# Multi-PPU
# ------------------------------------------------------------
NPROC_PER_NODE=16

# Do NOT hard-code CUDA_VISIBLE_DEVICES here.
# torchrun will use the 16 devices visible to this job/environment.

# ------------------------------------------------------------
# CPU / video decoding
#
# 128 CPU cores / 16 ranks:
#   4 decoder workers/rank = 64 PyAV workers total.
#
# This intentionally leaves CPU capacity for:
#   - 16 rank main processes
#   - dataset metadata / tensor assembly
#   - filesystem / I/O
#   - PyTorch runtime
#
# If profiling later shows queue wait is still high and storage is not
# saturated, try 6 workers/rank (96 total).
# ------------------------------------------------------------
export ZEVA_CACHE_VIDEO_BACKEND=pyav
export ZEVA_CACHE_DECODE_WORKERS=4
export ZEVA_CACHE_PREFETCH_EPISODES=8

# Keep intra-op thread pools small because we already have 16 ranks and
# 64 video decoder workers.
export ZEVA_CACHE_TORCH_CPU_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

# Optional profiling:
# export ZEVA_CACHE_PROFILE_GPU=1

# ------------------------------------------------------------
# Fixed Hydra configuration
# ------------------------------------------------------------
TASK="robotwin_zeva_fastwam_3cam_384"
CTE_CKPT="runs/zeva_cte/cte.pt"
CACHE_PATH="runs/zeva_cache/phase_effect_v4"

echo "============================================================"
echo "Zeva cache build"
echo "  PPU ranks           : ${NPROC_PER_NODE}"
echo "  decode workers/rank : ${ZEVA_CACHE_DECODE_WORKERS}"
echo "  total decoders      : $((NPROC_PER_NODE * ZEVA_CACHE_DECODE_WORKERS))"
echo "  prefetch/rank       : ${ZEVA_CACHE_PREFETCH_EPISODES}"
echo "  task                : ${TASK}"
echo "  CTE checkpoint      : ${CTE_CKPT}"
echo "  cache output        : ${CACHE_PATH}"
echo "============================================================"

exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/build_zeva_robotwin_cache.py \
  task="${TASK}" \
  model.zeva.cte.checkpoint="${CTE_CKPT}" \
  model.zeva.cache.path="${CACHE_PATH}"
