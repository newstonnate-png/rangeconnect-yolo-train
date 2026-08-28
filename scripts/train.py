#!/usr/bin/env python3
"""Train a YOLO bullet-hole detector and export it to ONNX.

Reads defaults from train_config.yaml (repo root); every key can be overridden
by an environment variable named after the upper-cased key (MODEL, IMGSZ,
EPOCHS, BATCH, PATIENCE, DEVICE, CACHE, RUN_NAME).

Runs inside the official `ultralytics/ultralytics` image on a Vast.ai GPU
instance, but works anywhere ultralytics is installed. Outputs land in
runs/<run_name>/ ; upload_artifacts.py ships them onward.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "train_config.yaml"
DATA_YAML = REPO_ROOT / "data" / "bullet-hole.yaml"

# key -> type coercion. Env var name is the upper-cased key.
_CASTS = {
    "model": str,
    "imgsz": int,
    "epochs": int,
    "batch": int,
    "patience": int,
    "device": str,
    "cache": str,
    "run_name": str,
}


def load_config() -> dict:
    cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    for key, cast in _CASTS.items():
        env_val = os.environ.get(key.upper())
        if env_val not in (None, ""):
            cfg[key] = env_val
        if cfg.get(key) is not None:
            cfg[key] = cast(cfg[key])
    return cfg


def main() -> int:
    if not DATA_YAML.exists():
        print(f"ERROR: {DATA_YAML} not found. Run fetch_dataset.py first.", file=sys.stderr)
        return 1

    cfg = load_config()
    print("=== resolved training config ===")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print("================================")

    model = YOLO(cfg["model"])

    # Small-object-friendly, orientation-invariant augmentation. Bullet holes
    # have no canonical "up", so vertical + horizontal flips are safe and
    # double the effective data. mosaic helps tiny objects; close it for the
    # last epochs so the model finishes on natural full-frame composition.
    results = model.train(
        data=str(DATA_YAML),
        imgsz=cfg["imgsz"],
        epochs=cfg["epochs"],
        batch=cfg["batch"],
        patience=cfg["patience"],
        device=cfg["device"],
        cache=cfg["cache"],
        project=str(REPO_ROOT / "runs"),
        name=cfg["run_name"],
        exist_ok=True,
        mosaic=1.0,
        close_mosaic=10,
        scale=0.5,
        fliplr=0.5,
        flipud=0.5,
        rect=False,
        seed=0,
    )

    save_dir = Path(results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    print(f"\nBest weights: {best_pt}")

    # opset 12 + no dynamic axes = widest onnxruntime compatibility on the
    # CPU-only scoring PC. simplify folds constant nodes for a leaner CPU graph.
    onnx_path = YOLO(str(best_pt)).export(
        format="onnx",
        imgsz=cfg["imgsz"],
        opset=12,
        simplify=True,
        dynamic=False,
    )
    print(f"ONNX: {onnx_path}")
    print(f"\nRun dir: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
