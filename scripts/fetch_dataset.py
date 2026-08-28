#!/usr/bin/env python3
"""Fetch and assemble the bullet-hole datasets on the training instance.

Two datasets are built:

  datasets/pool/      <- every source in PRETRAIN_SOURCES, merged. The generic
                         "what a bullet hole looks like" prior. Phase 1 trains here.
  datasets/finetune/  <- FINETUNE_SOURCE only (real zoom-lens frames on the
                         printed target). Phase 2 fine-tunes here. Optional —
                         omit for the first baseline run.

Each source is a spec (see scripts/sources.py):
  roboflow:<workspace>/<project>/<version>   (dataset export is free-tier)
  url:<https .zip/.tar.gz>
  hf:<dataset repo id>

Writes data/pool.yaml and (when present) data/finetune.yaml with absolute paths.
Any split with no val images gets one carved from train/ by make_val_split.py.

Env:
  PRETRAIN_SOURCES   comma-separated specs. If unset, falls back to a single
                     source built from the legacy ROBOFLOW_* / DATASET_SOURCE vars.
  FINETUNE_SOURCE    one spec, or empty.
  ROBOFLOW_API_KEY   required if any spec is roboflow:
  HF_TOKEN           required if any spec is hf:
  Legacy single-source fallback: DATASET_SOURCE (roboflow|hf), ROBOFLOW_WORKSPACE,
  ROBOFLOW_PROJECT, ROBOFLOW_VERSION, HF_DATASET_REPO.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import (  # noqa: E402
    count_images,
    download_source,
    merge_into,
    normalize_layout,
    write_data_yaml,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
POOL_DIR = REPO_ROOT / "datasets" / "pool"
FINETUNE_DIR = REPO_ROOT / "datasets" / "finetune"
SCRIPTS = REPO_ROOT / "scripts"


def legacy_pretrain_spec() -> str:
    """Reconstruct a single source spec from the old env vars so an existing
    setup keeps working without PRETRAIN_SOURCES."""
    src = os.environ.get("DATASET_SOURCE", "roboflow").lower()
    if src == "hf":
        repo = os.environ.get("HF_DATASET_REPO", "")
        if not repo:
            sys.exit("ERROR: DATASET_SOURCE=hf but HF_DATASET_REPO unset.")
        return f"hf:{repo}"
    ws = os.environ.get("ROBOFLOW_WORKSPACE", "justines-workspace-ls3un")
    proj = os.environ.get("ROBOFLOW_PROJECT", "justines-workspace-ls3un")
    ver = os.environ.get("ROBOFLOW_VERSION", "1")
    return f"roboflow:{ws}/{proj}/{ver}"


def build(dataset_dir: Path, specs: list[str], label: str) -> int:
    """Download each spec, normalize, merge into dataset_dir. Returns image count."""
    if dataset_dir.exists():
        # fresh each run so re-launches are deterministic
        import shutil

        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True)

    for i, spec in enumerate(specs):
        print(f"[{label}] source {i + 1}/{len(specs)}: {spec}")
        with tempfile.TemporaryDirectory(prefix=f"rc_{label}_{i}_") as tmp:
            raw = download_source(spec, Path(tmp))
            normalize_layout(raw)
            n = merge_into(raw, dataset_dir, prefix=f"s{i}")
            print(f"[{label}]   merged {n} images")

    total = count_images(dataset_dir)
    print(f"[{label}] total images: {total}")
    if total == 0:
        sys.exit(f"ERROR: {label} dataset is empty after fetch.")

    rc = subprocess.call([sys.executable, str(SCRIPTS / "make_val_split.py"),
                          "--base", str(dataset_dir)])
    if rc not in (0,):  # 1 = "nothing to split" is acceptable only if valid/ exists
        if count_images(dataset_dir / "valid" / "images") == 0:
            sys.exit(f"ERROR: {label} has no validation images and none could be split.")
    return total


def main() -> int:
    pretrain_env = os.environ.get("PRETRAIN_SOURCES", "").strip()
    pretrain_specs = (
        [s for s in pretrain_env.split(",") if s.strip()]
        if pretrain_env
        else [legacy_pretrain_spec()]
    )

    print("=== pretrain pool ===")
    build(POOL_DIR, pretrain_specs, "pool")
    write_data_yaml(DATA_DIR / "pool.yaml", POOL_DIR)
    # keep the legacy filename working as an alias for single-stage callers
    write_data_yaml(DATA_DIR / "bullet-hole.yaml", POOL_DIR)
    print(f"[pool] wrote {DATA_DIR / 'pool.yaml'}")

    finetune_spec = os.environ.get("FINETUNE_SOURCE", "").strip()
    if finetune_spec:
        print("\n=== fine-tune set ===")
        build(FINETUNE_DIR, [finetune_spec], "finetune")
        write_data_yaml(DATA_DIR / "finetune.yaml", FINETUNE_DIR)
        print(f"[finetune] wrote {DATA_DIR / 'finetune.yaml'}")
    else:
        # ensure a stale file from a previous run can't trigger phase 2
        (DATA_DIR / "finetune.yaml").unlink(missing_ok=True)
        print("\n[finetune] FINETUNE_SOURCE unset — phase 2 will be skipped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
