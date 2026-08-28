#!/usr/bin/env bash
# Vast.ai on-start provisioning for the RangeConnect YOLO training template.
# Runs inside the official `ultralytics/ultralytics:latest` image (torch +
# ultralytics preinstalled). Flow: deps -> fetch dataset -> train + export ONNX
# -> upload artifacts -> keep JupyterLab up for inspection.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

say() { echo "[provision] $*"; }

# Stable per-run stamp for artifact foldering. Exported so upload_artifacts.py
# reads the same value.
export RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
say "run stamp: $RUN_STAMP  |  model: ${MODEL:-<config default>}  |  dataset: ${DATASET_SOURCE:-roboflow}"

# ---- helper deps ----------------------------------------------------------
# The base image has torch + ultralytics. Add the dataset/artifact clients and
# JupyterLab for the keep-alive inspection shell. All three are light pure-python
# packages, so install unconditionally rather than sniffing the source specs.
say "installing helper deps"
pip install -q --no-input roboflow huggingface_hub jupyterlab || { say "pip install failed"; exit 1; }

# ---- dataset ------------------------------------------------------------------
# Builds datasets/pool/ from PRETRAIN_SOURCES and, if FINETUNE_SOURCE is set,
# datasets/finetune/ from that. Writes data/pool.yaml (+ data/finetune.yaml).
say "fetching datasets"
python scripts/fetch_dataset.py || { say "dataset fetch failed"; exit 1; }

# ---- train + export --------------------------------------------------------
# Phase 1 pretrains on the pool; phase 2 fine-tunes on the in-domain set when
# data/finetune.yaml exists; then the final best.pt is exported to ONNX.
say "starting training"
python scripts/train.py
train_rc=$?
if [[ $train_rc -ne 0 ]]; then
  say "training exited $train_rc — still attempting artifact upload of any partial run"
fi

# ---- upload -----------------------------------------------------------------
say "uploading artifacts"
python scripts/upload_artifacts.py || say "artifact upload failed (files remain under runs/)"

# ---- keep-alive for inspection --------------------------------------------
if [[ "${ENABLE_JUPYTER:-true}" =~ ^(1|true|yes|on)$ ]]; then
  if [[ -z "${JUPYTER_TOKEN:-}" ]]; then
    say "ENABLE_JUPYTER set but JUPYTER_TOKEN empty — refusing to start a tokenless public Jupyter"
    say "training done. Set JUPYTER_TOKEN and re-run, or pull files with 'vastai copy'."
    exit "$train_rc"
  fi
  say "training done — starting JupyterLab on :8888 for inspection"
  say "artifacts: runs/  (and pushed to HF_MODEL_REPO if set)"
  exec jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root \
    --ServerApp.token="$JUPYTER_TOKEN" --ServerApp.root_dir="$REPO_DIR"
fi

say "training done. runs/ holds the outputs. Remember to 'vastai destroy' this instance."
exit "$train_rc"
