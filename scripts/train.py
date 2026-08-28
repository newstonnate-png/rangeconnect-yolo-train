#!/usr/bin/env python3
"""Two-phase YOLO training for the RangeConnect bullet-hole detector, + ONNX export.

Phase 1  fine-tune the base weights (MODEL) on datasets/pool/ — the pooled
         public "what a hole looks like" prior.
Phase 2  if data/finetune.yaml exists, continue from phase-1 best.pt on
         datasets/finetune/ (real zoom-lens frames): fewer epochs, lower LR,
         backbone frozen. Skipped cleanly when there is no fine-tune data —
         that is the "baseline now" path.

Then export the final best.pt to ONNX (opset 12, static, simplified) for
onnxruntime CPU inference in GunRangeApp3/scoring_app.py.

Defaults come from train_config.yaml; every key is overridable by the
upper-cased env var (MODEL, IMGSZ, EPOCHS, BATCH, PATIENCE, DEVICE, CACHE,
RUN_NAME, FINETUNE_EPOCHS).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "train_config.yaml"
POOL_YAML = REPO_ROOT / "data" / "pool.yaml"
FINETUNE_YAML = REPO_ROOT / "data" / "finetune.yaml"
RUNS = REPO_ROOT / "runs"

_CASTS = {
    "model": str,
    "imgsz": int,
    "epochs": int,
    "batch": int,
    "patience": int,
    "device": str,
    "cache": str,
    "run_name": str,
    "finetune_epochs": int,
}

# Augmentation shared by both phases. Bullet holes have no canonical "up", so
# both flips are safe and double the effective data; mosaic helps tiny objects.
_AUG = dict(mosaic=1.0, close_mosaic=10, scale=0.5, fliplr=0.5, flipud=0.5, rect=False)


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
    if not POOL_YAML.exists():
        print(f"ERROR: {POOL_YAML} not found. Run fetch_dataset.py first.", file=sys.stderr)
        return 1

    cfg = load_config()
    print("=== resolved training config ===")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print("================================")

    # ---- phase 1: pooled pretrain ----
    print("\n### phase 1 — pretrain on datasets/pool/")
    p1 = YOLO(cfg["model"]).train(
        data=str(POOL_YAML),
        imgsz=cfg["imgsz"],
        epochs=cfg["epochs"],
        batch=cfg["batch"],
        patience=cfg["patience"],
        device=cfg["device"],
        cache=cfg["cache"],
        project=str(RUNS),
        name=f"{cfg['run_name']}-pretrain",
        exist_ok=True,
        seed=0,
        **_AUG,
    )
    best = Path(p1.save_dir) / "weights" / "best.pt"
    print(f"phase 1 best: {best}")

    # ---- phase 2: in-domain fine-tune ----
    if FINETUNE_YAML.exists():
        print("\n### phase 2 — fine-tune on datasets/finetune/")
        p2 = YOLO(str(best)).train(
            data=str(FINETUNE_YAML),
            imgsz=cfg["imgsz"],
            epochs=cfg["finetune_epochs"],
            batch=cfg["batch"],
            patience=max(10, cfg["finetune_epochs"] // 3),
            device=cfg["device"],
            cache=cfg["cache"],
            project=str(RUNS),
            name=f"{cfg['run_name']}-finetune",
            exist_ok=True,
            seed=0,
            lr0=0.002,      # ~1/5 of the default; the model is already close
            freeze=10,      # hold the backbone, adapt the head to this lens
            **_AUG,
        )
        best = Path(p2.save_dir) / "weights" / "best.pt"
        print(f"phase 2 best: {best}")
    else:
        print("\n### phase 2 skipped — no data/finetune.yaml (baseline run)")

    # ---- export ----
    print(f"\n### exporting ONNX from {best}")
    onnx_path = YOLO(str(best)).export(
        format="onnx", imgsz=cfg["imgsz"], opset=12, simplify=True, dynamic=False
    )
    print(f"ONNX: {onnx_path}")
    print(f"\nFinal weights: {best}")
    print(f"Run dir: {best.parent.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
