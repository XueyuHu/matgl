# CHGNet-OC20_mid

This folder stores a trained checkpoint for the plain CHGNet finetuning run (surface-aware disabled).

## Checkpoint

- `model.ckpt`: best validation checkpoint from run `oc22_chgnet_ablation_no_surface_8ep`
- Source file: `runs/oc22_chgnet_ablation_no_surface_8ep/checkpoints/surface-chgnet-epoch=07-val_Total_Loss=0.0135-v1.ckpt`

## Training Setup

- Dataset root: `/home/kaiwen/Desktop/SlabNet/oc22_aselmdb_mid`
- Train/Val size: `20000 / 4000`
- Epochs: `8`
- Batch size: `2` (val batch size `1`)
- LR: `5e-5`
- Precision: `32`
- Device: CUDA
- Surface-aware: disabled

## Final metrics (from lightning logs)

- `val_Total_Loss`: `0.013496`
- `val_Energy_MAE`: `0.028629`
- `val_Force_MAE`: `0.089008`

For additional run metadata, see `run_summary.json`.
