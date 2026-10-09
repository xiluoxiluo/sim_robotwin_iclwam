#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="/data/share/1919650160032350208/foundation_model/datasets/robotwin2.0"

cd "$ROOT"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/data/share/1919650160032350208/zjj/fastwam/checkpoints}"

# ============================================================
# Default paths
# 外部如果显式设置了环境变量，则优先使用外部值
# ============================================================

BASE_CKPT="${BASE_CKPT:-$ROOT/checkpoints/fastwam_ckpt.pt}"

CTE_CKPT="${CTE_CKPT:-$ROOT/runs/robotwin_zeva_fastwam_3cam_384/2026-09-23_20-32-07/cte.pt}"

PHASE_CACHE="${PHASE_CACHE:-$DATA/zeva_phase_effect_cache_v4}"

ART="${ART:-$DATA/zeva_stage2_artifacts}"

BEHAVIOR_BANK="${BEHAVIOR_BANK:-$ART/behavior_bank.pt}"
READOUT_CACHE="${READOUT_CACHE:-$ART/initial_readouts.pt}"
RETRIEVAL_HEAD="${RETRIEVAL_HEAD:-$ART/static_retrieval_head.pt}"
STATIC_CONTEXT_CACHE="${STATIC_CONTEXT_CACHE:-$ART/static_episode_context.pt}"
PIM_CACHE="${PIM_CACHE:-$ART/prepared_pim_v1}"

export BASE_CKPT
export CTE_CKPT
export PHASE_CACHE
export ART
export BEHAVIOR_BANK
export READOUT_CACHE
export RETRIEVAL_HEAD
export STATIC_CONTEXT_CACHE
export PIM_CACHE

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ============================================================
# Fail fast
# ============================================================

echo "============================================================"
echo " Zeva Stage-2 Prepared Validation"
echo "============================================================"
echo "ROOT                 = $ROOT"
echo "DATA                 = $DATA"
echo "BASE_CKPT            = $BASE_CKPT"
echo "CTE_CKPT             = $CTE_CKPT"
echo "PHASE_CACHE          = $PHASE_CACHE"
echo "BEHAVIOR_BANK        = $BEHAVIOR_BANK"
echo "READOUT_CACHE        = $READOUT_CACHE"
echo "RETRIEVAL_HEAD       = $RETRIEVAL_HEAD"
echo "STATIC_CONTEXT_CACHE = $STATIC_CONTEXT_CACHE"
echo "PIM_CACHE            = $PIM_CACHE"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "============================================================"

[[ -f "$BASE_CKPT" ]] || {
    echo "[ERROR] BASE_CKPT 不存在: $BASE_CKPT"
    exit 1
}

[[ -f "$CTE_CKPT" ]] || {
    echo "[ERROR] CTE_CKPT 不存在: $CTE_CKPT"
    exit 1
}

[[ -d "$PHASE_CACHE" ]] || {
    echo "[ERROR] PHASE_CACHE 不存在: $PHASE_CACHE"
    exit 1
}

[[ -f "$BEHAVIOR_BANK" ]] || {
    echo "[ERROR] BEHAVIOR_BANK 不存在: $BEHAVIOR_BANK"
    exit 1
}

[[ -f "$READOUT_CACHE" ]] || {
    echo "[ERROR] READOUT_CACHE 不存在: $READOUT_CACHE"
    exit 1
}

[[ -f "$RETRIEVAL_HEAD" ]] || {
    echo "[ERROR] RETRIEVAL_HEAD 不存在: $RETRIEVAL_HEAD"
    exit 1
}

[[ -f "$STATIC_CONTEXT_CACHE" ]] || {
    echo "[ERROR] STATIC_CONTEXT_CACHE 不存在: $STATIC_CONTEXT_CACHE"
    exit 1
}

[[ -d "$PIM_CACHE" ]] || {
    echo "[ERROR] PIM_CACHE 不存在: $PIM_CACHE"
    exit 1
}

# ============================================================
# Validation
# ============================================================

exec python scripts/validate_zeva_stage2_prepared.py \
  task=robotwin_zeva_fastwam_stage2_3cam_384 \
  ++ckpt="$BASE_CKPT" \
  ++model.zeva.cte.checkpoint="$CTE_CKPT" \
  ++model.zeva.cache.path="$PHASE_CACHE" \
  ++model.zeva.task_context.bank_path="$BEHAVIOR_BANK" \
  ++model.zeva.task_context.readout_cache_path="$READOUT_CACHE" \
  ++model.zeva.task_context.retrieval_checkpoint="$RETRIEVAL_HEAD" \
  ++model.zeva.task_context.static_episode_cache_path="$STATIC_CONTEXT_CACHE" \
  ++model.zeva.prepared_stage2.pim_cache_path="$PIM_CACHE" \
  "$@"