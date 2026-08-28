#!/usr/bin/env python3
"""Carve a deterministic validation split out of <base>/train/.

Used when a merged dataset has images only in train/ (the seed Roboflow export
has 1290 train images and 0 val). Moves a fixed fraction of image+label pairs
from train/ to valid/. Deterministic (sorted + fixed seed) so re-runs and
resumes are stable. No-op if valid/ already has images.

Usage:
  python make_val_split.py [--base datasets/pool]

Env:
  VAL_FRACTION  fraction of train/ to move to valid/  (default 0.15)
  SPLIT_SEED    RNG seed                               (default 0)
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def make_split(base: Path, fraction: float, seed: int) -> int:
    train_img, train_lbl = base / "train" / "images", base / "train" / "labels"
    val_img, val_lbl = base / "valid" / "images", base / "valid" / "labels"
    for d in (val_img, val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    if any(p.suffix.lower() in IMG_EXTS for p in val_img.iterdir()):
        print(f"make_val_split: {base}/valid already populated -- skipping")
        return 0

    images = sorted(p for p in train_img.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        print(f"make_val_split: no images in {base}/train — nothing to do")
        return 1

    rng = random.Random(seed)
    rng.shuffle(images)
    n_val = max(1, round(len(images) * fraction))

    moved = 0
    for img in images[:n_val]:
        lbl = train_lbl / f"{img.stem}.txt"
        shutil.move(str(img), str(val_img / img.name))
        if lbl.exists():
            shutil.move(str(lbl), str(val_lbl / lbl.name))
        moved += 1

    print(
        f"make_val_split: moved {moved}/{len(images)} pairs to {base}/valid "
        f"(fraction={fraction}, seed={seed})"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(REPO_ROOT / "datasets" / "pool"))
    args = ap.parse_args()
    fraction = float(os.environ.get("VAL_FRACTION", "0.15"))
    seed = int(os.environ.get("SPLIT_SEED", "0"))
    return make_split(Path(args.base), fraction, seed)


if __name__ == "__main__":
    raise SystemExit(main())
