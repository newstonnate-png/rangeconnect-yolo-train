#!/usr/bin/env python3
"""Carve a deterministic validation split out of datasets/bullet-hole/train/.

Used when the dataset export ships everything in train/ (the current Roboflow
export has 1290 train images and 0 val). Moves a fixed fraction of
image+label pairs from train/ to valid/. Deterministic (sorted + fixed seed)
so re-runs and resumes are stable.

Env:
  VAL_FRACTION  fraction of train/ to move to valid/  (default 0.15)
  SPLIT_SEED    RNG seed                               (default 0)
"""
from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE = REPO_ROOT / "datasets" / "bullet-hole"
VAL_FRACTION = float(os.environ.get("VAL_FRACTION", "0.15"))
SEED = int(os.environ.get("SPLIT_SEED", "0"))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    train_img = BASE / "train" / "images"
    train_lbl = BASE / "train" / "labels"
    val_img = BASE / "valid" / "images"
    val_lbl = BASE / "valid" / "labels"
    for d in (val_img, val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in train_img.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        print("make_val_split: no images in train/ — nothing to do")
        return 1

    rng = random.Random(SEED)
    rng.shuffle(images)
    n_val = max(1, round(len(images) * VAL_FRACTION))
    moving = images[:n_val]

    moved = 0
    for img in moving:
        lbl = train_lbl / f"{img.stem}.txt"
        shutil.move(str(img), str(val_img / img.name))
        if lbl.exists():
            shutil.move(str(lbl), str(val_lbl / lbl.name))
        moved += 1

    print(
        f"make_val_split: moved {moved}/{len(images)} pairs to valid/ "
        f"(fraction={VAL_FRACTION}, seed={SEED})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
