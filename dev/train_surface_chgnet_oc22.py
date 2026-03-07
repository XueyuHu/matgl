#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
from functools import partial
from pathlib import Path

import lightning as pl
import numpy as np
import torch
from ase import Atoms
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import Subset
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from pymatgen.io.ase import AseAtomsAdaptor

import matgl
from matgl.ext._pymatgen_dgl import Structure2Graph
from matgl.graph._data_dgl import MGLDataset, collate_fn_pes
from matgl.utils.training import PotentialLightningModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train surface-aware CHGNet on OC22 ASE-LMDB subset.")
    parser.add_argument("--data-root", type=Path, required=True, help="Root directory with train/val_id/val_ood *.aselmdb")
    parser.add_argument("--run-dir", type=Path, required=True, help="Output directory for logs/checkpoints/cache")
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="pretrained_models/CHGNet-MatPES-PBE-2025.2.10-2.7M-PES",
        help="MatGL pretrained model name or local path",
    )
    parser.add_argument("--max-train", type=int, default=20000, help="Max train structures")
    parser.add_argument("--max-val", type=int, default=4000, help="Max validation structures")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--val-batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1, help=">0 for quick smoke run")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--precision", choices=["32", "16-mixed", "bf16-mixed"], default="32")
    parser.add_argument("--surface-lambda1", type=float, default=0.5)
    parser.add_argument("--surface-lambda2", type=float, default=0.5)
    parser.add_argument("--surface-lambda3", type=float, default=0.5)
    parser.add_argument("--surface-lambda4", type=float, default=0.5)
    parser.add_argument("--adaptive-cutoff-alpha", type=float, default=0.35)
    parser.add_argument("--adaptive-cutoff-sharpness", type=float, default=8.0)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_lmdb_database_class() -> type:
    src = Path("/home/kaiwen/Desktop/SlabNet/mace/mace/tools/fairchem_dataset/lmdb_dataset_tools.py")
    spec = importlib.util.spec_from_file_location("lmdb_dataset_tools", src)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.LMDBDatabase


def iter_aselmdb_files(split_dir: Path) -> list[Path]:
    files = sorted(split_dir.glob("*.aselmdb"))
    if not files:
        raise FileNotFoundError(f"No *.aselmdb found under {split_dir}")
    return files


def load_split_structures(
    lmdb_cls: type,
    split_dirs: list[Path],
    limit: int,
) -> tuple[list, list[float], list[list[list[float]]]]:
    structures = []
    energies: list[float] = []
    forces: list[list[list[float]]] = []
    adaptor = AseAtomsAdaptor()

    for split_dir in split_dirs:
        for db_path in iter_aselmdb_files(split_dir):
            with lmdb_cls(db_path, readonly=True) as db:
                for idx in range(len(db.ids)):
                    row = db._get_row_by_index(idx, include_data=True)
                    row_arrays = row.data.get("__arrays__", {})

                    e = row.key_value_pairs.get("REF_energy", getattr(row, "REF_energy", None))
                    f = row_arrays.get("REF_forces")
                    if e is None or f is None:
                        continue

                    atoms = Atoms(
                        numbers=np.asarray(row.numbers, dtype=np.int32),
                        positions=np.asarray(row.positions, dtype=np.float64),
                        cell=np.asarray(row.cell, dtype=np.float64),
                        pbc=tuple(bool(x) for x in np.asarray(row.pbc).tolist()),
                    )
                    struct = adaptor.get_structure(atoms)

                    f_arr = np.asarray(f, dtype=np.float32)
                    if f_arr.shape != (len(struct), 3):
                        continue

                    structures.append(struct)
                    energies.append(float(e))
                    forces.append(f_arr.tolist())

                    if len(structures) >= limit:
                        return structures, energies, forces
    return structures, energies, forces


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MATGL_BACKEND", "DGL")
    os.environ.setdefault("MATGL_CACHE", str((args.run_dir / ".matgl_cache").resolve()))
    Path(os.environ["MATGL_CACHE"]).mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")

    lmdb_cls = load_lmdb_database_class()

    print(f"[INFO] Loading pretrained model: {args.pretrained_model}")
    pretrained = matgl.load_model(args.pretrained_model)
    model = pretrained.model
    element_refs = None
    if hasattr(pretrained, "element_refs") and hasattr(pretrained.element_refs, "property_offset"):
        element_refs = pretrained.element_refs.property_offset

    # Enable surface-aware path on the loaded CHGNet.
    model.surface_aware = True
    model.surface_lambda1 = args.surface_lambda1
    model.surface_lambda2 = args.surface_lambda2
    model.surface_lambda3 = args.surface_lambda3
    model.surface_lambda4 = args.surface_lambda4
    model.adaptive_receptive_field = True
    model.adaptive_cutoff_alpha = args.adaptive_cutoff_alpha
    model.adaptive_cutoff_sharpness = args.adaptive_cutoff_sharpness

    graph_cutoff = model.cutoff * (1.0 + model.adaptive_cutoff_alpha)
    converter = Structure2Graph(element_types=model.element_types, cutoff=graph_cutoff)

    train_dirs = [args.data_root / "train"]
    val_dirs = [args.data_root / "val_id", args.data_root / "val_ood"]

    print(f"[INFO] Reading train split from: {train_dirs}")
    train_structs, train_e, train_f = load_split_structures(lmdb_cls, train_dirs, args.max_train)
    print(f"[INFO] Reading val split from: {val_dirs}")
    val_structs, val_e, val_f = load_split_structures(lmdb_cls, val_dirs, args.max_val)
    print(f"[INFO] Loaded train={len(train_structs)}, val={len(val_structs)}")

    if len(train_structs) == 0 or len(val_structs) == 0:
        raise RuntimeError("Empty train/val set after loading. Check data paths and keys.")

    cache_dir = args.run_dir / "graph_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Always use dynamic line-graph construction in model forward.
    # This avoids compatibility failures between cached line-graphs and runtime graph state.
    include_line_graph = False
    print(f"[INFO] include_line_graph={include_line_graph} (dynamic line graph in CHGNet forward)")

    train_dataset = MGLDataset(
        include_line_graph=include_line_graph,
        converter=converter,
        threebody_cutoff=model.three_body_cutoff,
        directed_line_graph=include_line_graph,
        structures=train_structs,
        labels={"energies": train_e, "forces": train_f},
        directory_name=f"oc22_train_{len(train_structs)}",
        clear_processed=True,
        save_cache=True,
        raw_dir=str(cache_dir),
        save_dir=str(cache_dir),
    )
    val_dataset = MGLDataset(
        include_line_graph=include_line_graph,
        converter=converter,
        threebody_cutoff=model.three_body_cutoff,
        directed_line_graph=include_line_graph,
        structures=val_structs,
        labels={"energies": val_e, "forces": val_f},
        directory_name=f"oc22_val_{len(val_structs)}",
        clear_processed=True,
        save_cache=True,
        raw_dir=str(cache_dir),
        save_dir=str(cache_dir),
    )

    train_subset = Subset(train_dataset, list(range(len(train_dataset))))
    val_subset = Subset(val_dataset, list(range(len(val_dataset))))

    collate = partial(collate_fn_pes, include_stress=False, include_line_graph=include_line_graph)
    train_loader = GraphDataLoader(
        train_subset,
        shuffle=True,
        collate_fn=collate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=torch.Generator(device="cpu"),
    )
    val_loader = GraphDataLoader(
        val_subset,
        shuffle=False,
        collate_fn=collate,
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=torch.Generator(device="cpu"),
    )

    lit_module = PotentialLightningModule(
        model=model,
        include_line_graph=include_line_graph,
        stress_weight=0.0,
        lr=args.lr,
        energy_weight=1.0,
        force_weight=1.0,
        loss="huber_loss",
        element_refs=element_refs,
    )

    use_cuda = torch.cuda.is_available() and args.device in ("auto", "cuda")
    accelerator = "gpu" if use_cuda else "cpu"

    if args.max_steps > 0:
        ckpt_cb = ModelCheckpoint(
            dirpath=str(args.run_dir / "checkpoints"),
            filename="surface-chgnet-{epoch:02d}-{step}",
            save_top_k=0,
            save_last=True,
        )
    else:
        ckpt_cb = ModelCheckpoint(
            dirpath=str(args.run_dir / "checkpoints"),
            filename="surface-chgnet-{epoch:02d}-{val_Total_Loss:.4f}",
            monitor="val_Total_Loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    trainer_kwargs = dict(
        accelerator=accelerator,
        devices=1,
        max_epochs=args.epochs,
        precision=args.precision,
        gradient_clip_val=1.0,
        callbacks=[ckpt_cb, lr_cb],
        default_root_dir=str(args.run_dir),
        log_every_n_steps=20,
    )
    if args.max_steps > 0:
        trainer_kwargs["max_steps"] = args.max_steps

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    summary = {
        "pretrained_model": args.pretrained_model,
        "train_size": len(train_structs),
        "val_size": len(val_structs),
        "surface_aware": True,
        "graph_cutoff": graph_cutoff,
        "threebody_cutoff": model.three_body_cutoff,
        "run_dir": str(args.run_dir),
        "best_model_path": ckpt_cb.best_model_path,
    }
    with open(args.run_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[OK] Training finished.")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
