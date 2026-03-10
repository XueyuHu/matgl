# CHGNet-OC22-SurfaceAware-20k-v2

This folder stores a trained checkpoint for the surface-aware CHGNet finetuning run.

## Checkpoint

- `model.ckpt`: best validation checkpoint from run `oc22_surface_chgnet_20k_final_v2`
- Source file: `runs/oc22_surface_chgnet_20k_final_v2/checkpoints/surface-chgnet-epoch=07-val_Total_Loss=0.0151.ckpt`

## Training Setup

- Dataset root: `/home/kaiwen/Desktop/SlabNet/oc22_aselmdb_mid`
- Train/Val size: `20000 / 4000`
- Epochs: `8`
- Batch size: `2` (val batch size `1`)
- LR: `5e-5`
- Precision: `32`
- Device: CUDA
- Surface-aware: enabled

## Final metrics (from lightning logs)

- `val_Total_Loss`: `0.015053`
- `val_Energy_MAE`: `0.033122`
- `val_Force_MAE`: `0.093796`

For additional run metadata, see `run_summary.json`.
