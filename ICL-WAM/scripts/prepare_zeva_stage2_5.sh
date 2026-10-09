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
export CTE_CKPT="${CTE_CKPT:-/data/share/1919650160032350208/zjj/ICL-WAM/runs/robotwin_zeva_fastwam_3cam_384/2026-09-23_20-32-07/cte.pt}"
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
export GPU_IDS="${GPU_IDS:-0}"

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
echo "========== [5/5] Stage-2 Prepared Validation =========="
CUDA_VISIBLE_DEVICES=0 \
ZEVA_STAGE2_VALIDATE_SAMPLES="$ZEVA_STAGE2_VALIDATE_SAMPLES" \
ZEVA_STAGE2_MODEL_SMOKE="$ZEVA_STAGE2_MODEL_SMOKE" \
ZEVA_STAGE2_BACKWARD_SMOKE="$ZEVA_STAGE2_BACKWARD_SMOKE" \
bash scripts/validate_zeva_stage2_prepared.sh