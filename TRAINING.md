# Training Notes (Surface-Aware CHGNet)

This file summarizes the **working training setup** used in this repository.

## Environment (validated)

- Python: 3.12
- torch: 2.9.1+cu128
- torchvision: 0.24.1
- torchaudio: 2.9.1
- dgl: 2.1.0+cu121 (CUDA build)
- matgl: 2.0.6
- lightning: 2.6.0

## Data

Current OC22 subset used for finetuning:

- Train: `20000`
- Val: `4000` (`val_id + val_ood`)
- Root: `/home/kaiwen/Desktop/SlabNet/oc22_aselmdb_mid`

## Main Training Script

- `dev/train_surface_chgnet_oc22.py`
- Launcher: `train_surface_chgnet_oc22.sh`

## Recommended Stable Command

```bash
cd /home/kaiwen/Desktop/SlabNet/matgl_surface_aware/matgl
source .venv/bin/activate

PYTORCH_ALLOC_CONF=expandable_segments:True \
MATGL_BACKEND=DGL \
DATA_ROOT=/home/kaiwen/Desktop/SlabNet/oc22_aselmdb_mid \
RUN_NAME=oc22_surface_chgnet_20k_final_v2 \
DEVICE=cuda \
EPOCHS=8 \
MAX_STEPS=-1 \
BATCH_SIZE=2 \
VAL_BATCH_SIZE=1 \
NUM_WORKERS=0 \
PRECISION=32 \
LR=5e-5 \
./train_surface_chgnet_oc22.sh
```

## Known Stability Decisions

- Dynamic line graph is used in model forward (no precomputed line graph input in dataloader path).
- Surface descriptor path is stabilized to reduce NaN risk in training.
- Gradient clipping is enabled.
- `NUM_WORKERS=0` is recommended in this machine setup to avoid multiprocessing semaphore permission issues.

## Quick Smoke Run

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
MATGL_BACKEND=DGL \
DATA_ROOT=/home/kaiwen/Desktop/SlabNet/oc22_aselmdb_mid \
RUN_NAME=oc22_surface_chgnet_gpu_smoke \
DEVICE=cuda \
EPOCHS=1 \
MAX_STEPS=300 \
BATCH_SIZE=2 \
VAL_BATCH_SIZE=1 \
NUM_WORKERS=0 \
PRECISION=32 \
LR=5e-5 \
./train_surface_chgnet_oc22.sh
```

This smoke run configuration has been observed to complete stably without NaN.
