#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

source .venv/bin/activate
export MATGL_BACKEND="${MATGL_BACKEND:-DGL}"
export MATGL_CACHE="${MATGL_CACHE:-${ROOT_DIR}/.matgl_cache}"
mkdir -p "${MATGL_CACHE}"

# Ensure CUDA shared libraries bundled by pip packages are discoverable by DGL.
CUDA_LIB_DIRS=(
  "${ROOT_DIR}/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
  "${ROOT_DIR}/.venv/lib/python3.12/site-packages/nvidia/cublas/lib"
  "${ROOT_DIR}/.venv/lib/python3.12/site-packages/nvidia/cusparse/lib"
  "${ROOT_DIR}/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
  "${ROOT_DIR}/.venv/lib/python3.12/site-packages/nvidia/cusolver/lib"
  "${ROOT_DIR}/.venv/lib/python3.12/site-packages/nvidia/nccl/lib"
)
for d in "${CUDA_LIB_DIRS[@]}"; do
  if [[ -d "${d}" ]]; then
    export LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH:-}"
  fi
done

DATA_ROOT="${DATA_ROOT:-/home/kaiwen/Desktop/SlabNet/oc22_aselmdb_mid}"
RUN_NAME="${RUN_NAME:-oc22_surface_chgnet_mid_20k}"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/runs/${RUN_NAME}}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-pretrained_models/CHGNet-MatPES-PBE-2025.2.10-2.7M-PES}"

MAX_TRAIN="${MAX_TRAIN:-20000}"
MAX_VAL="${MAX_VAL:-4000}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:--1}"
BATCH_SIZE="${BATCH_SIZE:-6}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-6}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-1e-4}"
DEVICE="${DEVICE:-auto}"
PRECISION="${PRECISION:-32}"

echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] RUN_DIR=${RUN_DIR}"
echo "[INFO] PRETRAINED_MODEL=${PRETRAINED_MODEL}"
echo "[INFO] MAX_TRAIN=${MAX_TRAIN}, MAX_VAL=${MAX_VAL}"
echo "[INFO] EPOCHS=${EPOCHS}, MAX_STEPS=${MAX_STEPS}"
echo "[INFO] BATCH_SIZE=${BATCH_SIZE}, VAL_BATCH_SIZE=${VAL_BATCH_SIZE}, NUM_WORKERS=${NUM_WORKERS}"

python dev/train_surface_chgnet_oc22.py \
  --data-root "${DATA_ROOT}" \
  --run-dir "${RUN_DIR}" \
  --pretrained-model "${PRETRAINED_MODEL}" \
  --max-train "${MAX_TRAIN}" \
  --max-val "${MAX_VAL}" \
  --batch-size "${BATCH_SIZE}" \
  --val-batch-size "${VAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --epochs "${EPOCHS}" \
  --max-steps "${MAX_STEPS}" \
  --lr "${LR}" \
  --device "${DEVICE}" \
  --precision "${PRECISION}"
