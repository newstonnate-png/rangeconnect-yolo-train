#!/usr/bin/env python3
"""Push training artifacts to a private HuggingFace *model* repo.

Uploads best.pt, best.onnx, and the run's plots/metrics into a
timestamp+model-tagged subfolder so multiple runs coexist. Auth via HF_TOKEN.
No-op with a clear message if HF_MODEL_REPO is unset (grab from JupyterLab
instead).

Env:
  HF_MODEL_REPO   e.g. "newstonnate/rc-bullet-hole-yolo" (created private if absent)
  HF_TOKEN        write token
  MODEL           base model name, used only for the subfolder label
  RUN_STAMP       per-run stamp for the subfolder (set by provision.sh)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"

ARTIFACT_NAMES = [
    "weights/best.pt",
    "weights/best.onnx",
    "results.png",
    "results.csv",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "args.yaml",
    "PR_curve.png",
    "labels.jpg",
]


def latest_run() -> "Path | None":
    if not RUNS_DIR.exists():
        return None
    runs = [p for p in RUNS_DIR.iterdir() if p.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)


def main() -> int:
    repo = os.environ.get("HF_MODEL_REPO")
    if not repo:
        print(
            "upload_artifacts: HF_MODEL_REPO unset — skipping upload. "
            "Grab files from JupyterLab / 'vastai copy' instead."
        )
        return 0

    run = latest_run()
    if run is None:
        print("ERROR: no run directory under runs/.", file=sys.stderr)
        return 1

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    model = os.environ.get("MODEL", "yolo").replace(".pt", "")
    stamp = os.environ.get("RUN_STAMP", "run")
    subdir = f"{stamp}-{model}"

    api = HfApi(token=token)
    api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)

    uploaded = 0
    for rel in ARTIFACT_NAMES:
        src = run / rel
        if not src.exists():
            continue
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=f"{subdir}/{Path(rel).name}",
            repo_id=repo,
            repo_type="model",
            token=token,
        )
        print(f"  uploaded {rel} -> {subdir}/{Path(rel).name}")
        uploaded += 1

    if uploaded == 0:
        print("ERROR: found the run dir but no known artifacts in it.", file=sys.stderr)
        return 1

    print(
        f"upload_artifacts: {uploaded} files -> "
        f"https://huggingface.co/{repo}/tree/main/{subdir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
