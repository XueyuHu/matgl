#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check CHGNet training environment.")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Requested runtime device",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    os.environ.setdefault("MATGL_BACKEND", "DGL")
    os.environ.setdefault("MATGL_CACHE", str(repo_root / ".matgl_cache"))
    Path(os.environ["MATGL_CACHE"]).mkdir(parents=True, exist_ok=True)

    import torch
    import dgl
    import matgl

    cuda_ok = torch.cuda.is_available()
    if args.device == "cuda" and not cuda_ok:
        print("[ERROR] --device=cuda but torch.cuda.is_available() is False")
        return 2

    device = "cuda" if (args.device in ("auto", "cuda") and cuda_ok) else "cpu"
    print(f"[INFO] torch={torch.__version__} (cuda={torch.version.cuda})")
    print(f"[INFO] dgl={dgl.__version__}")
    print(f"[INFO] matgl={matgl.__version__}")
    print(f"[INFO] MATGL_BACKEND={os.environ['MATGL_BACKEND']}")
    print(f"[INFO] MATGL_CACHE={os.environ['MATGL_CACHE']}")
    print(f"[INFO] torch.cuda.is_available()={cuda_ok}, device={device}")

    # Validate CHGNet checkpoint loading.
    _ = matgl.load_model("pretrained_models/CHGNet-MPtrj-2023.12.1-2.7M-PES")
    print("[OK] CHGNet pretrained model loaded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
