#!/usr/bin/env bash
set -euo pipefail

cd /data/share/1919650160032350208/zjj/ICL-WAM

export PYTHONPATH=/data/share/1919650160032350208/zjj/ICL-WAM/src:${PYTHONPATH:-}
export DIFFSYNTH_MODEL_BASE_PATH=/data/share/1919650160032350208/zjj/fastwam/checkpoints

TASK="robotwin_zeva_fastwam_3cam_384"
CTE_CKPT="runs/robotwin_zeva_fastwam_3cam_384/2026-09-23_20-32-07/cte.pt"
LATENT_CACHE="/data/share/1919650160032350208/foundation_model/datasets/robotwin2.0/zeva_cte_latent_cache_v1"
PHASE_CACHE="/data/share/1919650160032350208/foundation_model/datasets/robotwin2.0/zeva_phase_effect_cache_v4"

# Deep recomputation is intentionally single-GPU. Set DEEP_EPISODES=0 for a
# cache-only structural/statistical pass; 8 is a good formal default.
DEEP_EPISODES="${DEEP_EPISODES:-8}"
SAMPLE_PHASE="${SAMPLE_PHASE:-4096}"
SAMPLE_EFFECT="${SAMPLE_EFFECT:-4096}"
PAIR_SAMPLES="${PAIR_SAMPLES:-20000}"
PROGRESSION_EPISODES="${PROGRESSION_EPISODES:-64}"
ADJACENT_EFFECT_PAIRS="${ADJACENT_EFFECT_PAIRS:-2048}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python scripts/validate_zeva_phase_effect_cache.py \
  task="${TASK}" \
  ++model.zeva.cte.checkpoint="${CTE_CKPT}" \
  ++model.zeva.cte.latent_cache_path="${LATENT_CACHE}" \
  ++model.zeva.cache.path="${PHASE_CACHE}" \
  ++phase_effect_validation.sample_phase="${SAMPLE_PHASE}" \
  ++phase_effect_validation.sample_effect="${SAMPLE_EFFECT}" \
  ++phase_effect_validation.pair_samples="${PAIR_SAMPLES}" \
  ++phase_effect_validation.progression_episodes="${PROGRESSION_EPISODES}" \
  ++phase_effect_validation.adjacent_effect_pairs="${ADJACENT_EFFECT_PAIRS}" \
  ++phase_effect_validation.deep_episodes="${DEEP_EPISODES}" \
  ++phase_effect_validation.deep_atol=1e-5 \
  ++phase_effect_validation.deep_rtol=1e-5 \
  ++phase_effect_validation.seed=20260924
