#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${1:?Usage: bash scripts/train_zeva_cte.sh <nproc_per_node> [hydra_overrides...]}"
shift

if ! [[ "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]]; then
  echo "Error: nproc_per_node must be a positive integer, got '${NPROC_PER_NODE}'." >&2
  exit 1
fi
if (( NPROC_PER_NODE < 1 )); then
  echo "Error: nproc_per_node must be at least 1." >&2
  exit 1
fi

cd /data/share/1919650160032350208/zjj/ICL-WAM
export DIFFSYNTH_MODEL_BASE_PATH="/data/share/1919650160032350208/zjj/fastwam/checkpoints"
export PYTHONPATH="/data/share/1919650160032350208/zjj/ICL-WAM/src:${PYTHONPATH:-}"

EXTRA_ARGS=("$@")
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_MACHINES="${NNODES:-1}"
MACHINE_RANK="${NODE_RANK:-0}"
MAIN_PROCESS_IP="${MASTER_ADDR:-127.0.0.1}"
MAIN_PROCESS_PORT="${MASTER_PORT:-29500}"

is_integer() {
  [[ "${1}" =~ ^[0-9]+$ ]]
}

if ! is_integer "${NUM_MACHINES}" || (( NUM_MACHINES < 1 )); then
  echo "Error: NNODES (${NUM_MACHINES}) must be a positive integer." >&2
  exit 1
fi
if ! is_integer "${MACHINE_RANK}" || (( MACHINE_RANK < 0 || MACHINE_RANK >= NUM_MACHINES )); then
  echo "Error: NODE_RANK (${MACHINE_RANK}) must be in [0, ${NUM_MACHINES})." >&2
  exit 1
fi
if ! is_integer "${MAIN_PROCESS_PORT}" || (( MAIN_PROCESS_PORT < 1 || MAIN_PROCESS_PORT > 65535 )); then
  echo "Error: MASTER_PORT (${MAIN_PROCESS_PORT}) must be a valid TCP port." >&2
  exit 1
fi

TASK_BASENAME="zeva_cte"

extract_task_basename() {
  local cfg="$1"
  if [[ "${cfg}" == task/* ]]; then
    local name="${cfg#task/}"
    name="${name%.yaml}"
    echo "${name}"
    return 0
  fi
  return 1
}

for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  arg="${EXTRA_ARGS[$i]}"
  case "${arg}" in
    --config-name)
      if ((i + 1 < ${#EXTRA_ARGS[@]})); then
        next="${EXTRA_ARGS[$((i + 1))]}"
        if parsed="$(extract_task_basename "${next}")"; then
          TASK_BASENAME="${parsed}"
        fi
      fi
      ;;
    --config-name=*)
      cfg="${arg#--config-name=}"
      if parsed="$(extract_task_basename "${cfg}")"; then
        TASK_BASENAME="${parsed}"
      fi
      ;;
    task=*)
      cfg="${arg#task=}"
      cfg="${cfg%.yaml}"
      TASK_BASENAME="${cfg}"
      ;;
  esac
done

if [[ -z "${RUN_ID:-}" ]]; then
  if (( NUM_MACHINES <= 1 )); then
    RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
  else
    RUN_ID_SYNC_TIMEOUT="${RUN_ID_SYNC_TIMEOUT:-180}"
    RUN_ID_SYNC_PORT="${RUN_ID_SYNC_PORT:-$((MAIN_PROCESS_PORT + 11))}"
    export RUN_ID_SYNC_HOST="${MAIN_PROCESS_IP}"
    export RUN_ID_SYNC_PORT
    export RUN_ID_SYNC_TIMEOUT
    export RUN_ID_SYNC_MACHINE_RANK="${MACHINE_RANK}"
    export RUN_ID_SYNC_NUM_MACHINES="${NUM_MACHINES}"
    export RUN_ID_SYNC_TASK_BASENAME="${TASK_BASENAME}"
    RUN_ID="$("${PYTHON_BIN}" - <<'PY'
import datetime
import os
from datetime import timedelta

import torch.distributed as dist

store = dist.TCPStore(
    host_name=os.environ["RUN_ID_SYNC_HOST"],
    port=int(os.environ["RUN_ID_SYNC_PORT"]),
    world_size=int(os.environ["RUN_ID_SYNC_NUM_MACHINES"]),
    is_master=int(os.environ["RUN_ID_SYNC_MACHINE_RANK"]) == 0,
    timeout=timedelta(seconds=int(os.environ["RUN_ID_SYNC_TIMEOUT"])),
)
key = f"run_id::{os.environ.get('RUN_ID_SYNC_TASK_BASENAME', 'zeva_cte')}"
if int(os.environ["RUN_ID_SYNC_MACHINE_RANK"]) == 0:
    store.set(key, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
print(store.get(key).decode("utf-8"))
PY
    )"
    echo "[run_id_sync] host=${RUN_ID_SYNC_HOST} port=${RUN_ID_SYNC_PORT} run_id=${RUN_ID}"
  fi
fi
OUTPUT_DIR="./runs/${TASK_BASENAME}/${RUN_ID}"

# Let an explicit Hydra output_dir override the generated run directory.
for arg in "${EXTRA_ARGS[@]}"; do
  case "${arg}" in
    output_dir=*|+output_dir=*|++output_dir=*)
      OUTPUT_DIR=""
      break
      ;;
  esac
done

echo "[launch] nproc_per_node=${NPROC_PER_NODE} run_id=${RUN_ID}"
if [[ -n "${OUTPUT_DIR}" ]]; then
  echo "[launch] output_dir=${OUTPUT_DIR}"
  EXTRA_ARGS=("output_dir=${OUTPUT_DIR}" "${EXTRA_ARGS[@]}")
else
  EXTRA_ARGS=("${EXTRA_ARGS[@]}")
fi

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --nnodes "${NUM_MACHINES}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --node_rank "${MACHINE_RANK}" \
  --master_addr "${MAIN_PROCESS_IP}" \
  --master_port "${MAIN_PROCESS_PORT}" \
  scripts/train_zeva_cte.py \
  "${EXTRA_ARGS[@]}"
