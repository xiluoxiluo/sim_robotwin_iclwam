#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# ICL-WAM Zeva -> FastWAM
# 一键完成 Stage-2 正式训练前的全部离线准备与验证
#
# 顺序：
#   A. Behavior Bank + Initial Readouts        (16 GPU)
#   B. Static Retrieval Head                   (1 GPU)
#   C. Static Episode Context Cache            (1 GPU)
#   D. Prepared PIM/BIT Cache                  (16 GPU)
#   E. Stage-2 Forward/Backward Smoke Test     (1 GPU)
#
# 注意：
#   本脚本默认不会启动正式 Stage-2 训练。
#   所有验证通过后，再单独启动 train_zeva_fastwam_prepared.sh。
# ============================================================

ROOT="${ROOT:-/data/share/1919650160032350208/zjj/ICL-WAM}"
DATA="${DATA:-/data/share/1919650160032350208/foundation_model/datasets/robotwin2.0}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/data/share/1919650160032350208/zjj/fastwam/checkpoints}"

# ------------------------------------------------------------
# 只需要你确认这一项。
# 推荐运行时传入：
BASE_CKPT=/data/share/1919650160032350208/zjj/ICL-WAM/checkpoints/fastwam_ckpt.pt
# ------------------------------------------------------------

# 已完成并验证的 Stage-1 / cache
export CTE_CKPT="${CTE_CKPT:-$ROOT/runs/robotwin_zeva_fastwam_3cam_384/2026-09-23_20-32-07/cte.pt}"
export LATENT_CACHE="${LATENT_CACHE:-$DATA/zeva_cte_latent_cache_v1}"
export PHASE_CACHE="${PHASE_CACHE:-$DATA/zeva_phase_effect_cache_v4}"

# Stage-2 artifacts
export ART="${ART:-$DATA/zeva_stage2_artifacts}"
mkdir -p "$ART"

export BEHAVIOR_BANK="${BEHAVIOR_BANK:-$ART/behavior_bank.pt}"
export READOUT_CACHE="${READOUT_CACHE:-$ART/initial_readouts.pt}"
export RETRIEVAL_HEAD="${RETRIEVAL_HEAD:-$ART/static_retrieval_head.pt}"
export STATIC_CONTEXT_CACHE="${STATIC_CONTEXT_CACHE:-$ART/static_episode_context.pt}"
export PIM_CACHE="${PIM_CACHE:-$ART/prepared_pim_v1}"

# 16 GPU
export GPU_IDS="${GPU_IDS:-0,1}"

# Conservative defaults
export ZEVA_BEHAVIOR_READOUT_BATCH_SIZE="${ZEVA_BEHAVIOR_READOUT_BATCH_SIZE:-2}"
export ZEVA_BEHAVIOR_READOUT_WORKERS="${ZEVA_BEHAVIOR_READOUT_WORKERS:-2}"
export ZEVA_BEHAVIOR_TAIL_BATCH_SIZE="${ZEVA_BEHAVIOR_TAIL_BATCH_SIZE:-4}"
export ZEVA_BEHAVIOR_TAIL_WORKERS="${ZEVA_BEHAVIOR_TAIL_WORKERS:-2}"
export ZEVA_STATIC_CONTEXT_BATCH_SIZE="${ZEVA_STATIC_CONTEXT_BATCH_SIZE:-512}"
export ZEVA_PIM_QUERY_BATCH="${ZEVA_PIM_QUERY_BATCH:-1024}"
export ZEVA_STAGE2_VALIDATE_SAMPLES="${ZEVA_STAGE2_VALIDATE_SAMPLES:-64}"
export ZEVA_STAGE2_MODEL_SMOKE="${ZEVA_STAGE2_MODEL_SMOKE:-1}"
export ZEVA_STAGE2_BACKWARD_SMOKE="${ZEVA_STAGE2_BACKWARD_SMOKE:-1}"

echo "============================================================"
echo " Zeva Stage-2 preparation"
echo "============================================================"
echo "ROOT              = $ROOT"
echo "DATA              = $DATA"
echo "BASE_CKPT         = $BASE_CKPT"
echo "CTE_CKPT          = $CTE_CKPT"
echo "LATENT_CACHE      = $LATENT_CACHE"
echo "PHASE_CACHE       = $PHASE_CACHE"
echo "ART               = $ART"
echo "GPU_IDS           = $GPU_IDS"
echo "============================================================"

# ------------------------------------------------------------
# Fail-fast：确认关键输入
# ------------------------------------------------------------
[[ -f "$BASE_CKPT" ]] || { echo "[ERROR] BASE_CKPT 不存在: $BASE_CKPT"; exit 1; }
[[ -f "$CTE_CKPT" ]] || { echo "[ERROR] CTE_CKPT 不存在: $CTE_CKPT"; exit 1; }
[[ -d "$LATENT_CACHE" ]] || { echo "[ERROR] LATENT_CACHE 不存在: $LATENT_CACHE"; exit 1; }
[[ -d "$PHASE_CACHE" ]] || { echo "[ERROR] PHASE_CACHE 不存在: $PHASE_CACHE"; exit 1; }

# ------------------------------------------------------------
# 0. 静态检查
# ------------------------------------------------------------
echo
echo "========== [0/5] Python static check =========="
python -m py_compile \
  src/fastwam/zeva/cte_latent_cache.py \
  src/fastwam/zeva/prepared_stage2.py \
  scripts/build_zeva_behavior_bank_optimized.py \
  scripts/train_zeva_task_context_retrieval_optimized.py \
  scripts/build_zeva_static_episode_context_cache.py \
  scripts/build_zeva_pim_retrieval_cache.py \
  scripts/validate_zeva_stage2_prepared.py \
  scripts/train_zeva_fastwam_prepared.py

# ------------------------------------------------------------
# A. Behavior Bank + Initial Readouts
# ------------------------------------------------------------
echo
echo "========== [1/5] Behavior Bank + Initial Readouts =========="
GPU_IDS="$GPU_IDS" \
ZEVA_BEHAVIOR_READOUT_BATCH_SIZE="$ZEVA_BEHAVIOR_READOUT_BATCH_SIZE" \
ZEVA_BEHAVIOR_READOUT_WORKERS="$ZEVA_BEHAVIOR_READOUT_WORKERS" \
ZEVA_BEHAVIOR_TAIL_BATCH_SIZE="$ZEVA_BEHAVIOR_TAIL_BATCH_SIZE" \
ZEVA_BEHAVIOR_TAIL_WORKERS="$ZEVA_BEHAVIOR_TAIL_WORKERS" \
bash scripts/build_zeva_behavior_bank_optimized.sh

[[ -f "$BEHAVIOR_BANK" ]] || { echo "[ERROR] Behavior bank 未生成"; exit 1; }
[[ -f "$READOUT_CACHE" ]] || { echo "[ERROR] Initial readout cache 未生成"; exit 1; }

# ------------------------------------------------------------
# B. Static Retrieval Head
# ------------------------------------------------------------
echo
echo "========== [2/5] Static Retrieval Head =========="
CUDA_VISIBLE_DEVICES=0 \
bash scripts/train_zeva_task_context_retrieval_optimized.sh

[[ -f "$RETRIEVAL_HEAD" ]] || { echo "[ERROR] Retrieval head 未生成"; exit 1; }

# ------------------------------------------------------------
# C. Static Episode Context Cache
# ------------------------------------------------------------
echo
echo "========== [3/5] Static Episode Context Cache =========="
CUDA_VISIBLE_DEVICES=0 \
ZEVA_STATIC_CONTEXT_BATCH_SIZE="$ZEVA_STATIC_CONTEXT_BATCH_SIZE" \
bash scripts/build_zeva_static_episode_context_cache.sh

[[ -f "$STATIC_CONTEXT_CACHE" ]] || { echo "[ERROR] Static episode context cache 未生成"; exit 1; }

# ------------------------------------------------------------
# D. Prepared PIM/BIT Cache
# ------------------------------------------------------------
echo
echo "========== [4/5] Prepared PIM/BIT Cache =========="
GPU_IDS="$GPU_IDS" \
ZEVA_PIM_QUERY_BATCH="$ZEVA_PIM_QUERY_BATCH" \
bash scripts/build_zeva_pim_retrieval_cache.sh

[[ -d "$PIM_CACHE" ]] || { echo "[ERROR] Prepared PIM cache 未生成"; exit 1; }

# ------------------------------------------------------------
# E. Stage-2 总验证：真实 Frozen FastWAM forward + backward
# ------------------------------------------------------------
echo
echo "========== [5/5] Stage-2 Prepared Validation =========="
CUDA_VISIBLE_DEVICES=0 \
ZEVA_STAGE2_VALIDATE_SAMPLES="$ZEVA_STAGE2_VALIDATE_SAMPLES" \
ZEVA_STAGE2_MODEL_SMOKE="$ZEVA_STAGE2_MODEL_SMOKE" \
ZEVA_STAGE2_BACKWARD_SMOKE="$ZEVA_STAGE2_BACKWARD_SMOKE" \
bash scripts/validate_zeva_stage2_prepared.sh

echo
echo "============================================================"
echo " STAGE-2 PREPARATION COMPLETE"
echo "============================================================"
echo "Behavior bank       : $BEHAVIOR_BANK"
echo "Initial readouts    : $READOUT_CACHE"
echo "Retrieval head      : $RETRIEVAL_HEAD"
echo "Static context      : $STATIC_CONTEXT_CACHE"
echo "Prepared PIM/BIT    : $PIM_CACHE"
echo
echo "如果上面同时出现："
echo "  PREPARED STAGE-2 VALIDATION: PASSED"
echo "  FROZEN FASTWAM + ZEVA STAGE-2 FORWARD/BACKWARD: PASSED"
echo
echo "则可以开始正式 Stage-2A："
echo
echo "GPU_IDS=$GPU_IDS bash scripts/train_zeva_fastwam_prepared.sh"
echo "============================================================"
