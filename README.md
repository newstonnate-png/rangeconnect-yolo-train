# rangeconnect-yolo-train

Vast.ai training template for the RangeConnect bullet-hole detector. Rents a GPU,
assembles the dataset(s), trains a YOLO model in two phases, exports ONNX, and
pushes the weights to a private HuggingFace repo. Inference then runs 100 % on
CPU in `GunRangeApp3/scoring_app.py`.

**No image to build.** It rides the official `ultralytics/ultralytics:latest`
image plus `scripts/provision.sh`.

## The two-phase idea

The detector only ever learns "hole / not hole" — one class, no zones, no colour,
no score (zone scoring is a separate colour-sampling step in `scoring_core.py`).
So dataset choice only affects **hole localization**, and no public dataset
matches *this silhouette target, through this camera's zoom lens, at this lane
distance*. Hence:

1. **Phase 1 — pretrain** on `datasets/pool/`: four free public (CC BY 4.0)
   Roboflow Universe bullet-hole sets, merged and folded to one class. A robust
   generic prior. See `Template/rangeconnect-yolo-train.md` for the list.
2. **Phase 2 — fine-tune** on `datasets/finetune/`: 200–400 real zoom-lens frames
   on the printed target (captured with `GunRangeApp3/camera_view.py`, labelled
   in Roboflow as a new dataset version). Fewer epochs, lower LR, frozen
   backbone. **Skipped cleanly when there is no fine-tune set** — that is the
   first "baseline" run.

## What you need

- A Vast.ai account + `vastai` CLI.
- A **Roboflow** account (API key is free; dataset export is a free-tier
  feature — hosted inference and Roboflow Train are the paid parts and are not
  used here).
- A **HuggingFace** token + a private model repo name for the artifacts.

## Run a training job

1. Fill real values for `ROBOFLOW_API_KEY`, `HF_TOKEN`, `HF_MODEL_REPO`,
   `JUPYTER_TOKEN` (kept in `.env`, gitignored) into the
   `Template/rangeconnect-yolo-train.md` one-liner. `PRETRAIN_SOURCES` is already
   the public pool; leave `FINETUNE_SOURCE=` empty for the first run.
2. Create the Vast instance: image `ultralytics/ultralytics:latest`, launch mode
   **docker ENTRYPOINT**, 40 GB disk, one RTX 3090/5090. Paste the on-start
   script and the env one-liner.
3. `scripts/provision.sh` runs: `deps → fetch_dataset.py → train.py (phase 1
   [+ phase 2] + ONNX export) → upload_artifacts.py → JupyterLab on :8888`.
4. When `results.png` looks converged and `best.onnx` is in the HF model repo,
   `vastai destroy` the instance.
5. Later: capture + label real frames, publish them as Roboflow **v2**, re-run
   with `-e FINETUNE_SOURCE=roboflow:<ws>/<proj>/2`.

## Config

`train_config.yaml` holds the defaults; any env var (`MODEL`, `IMGSZ`, `EPOCHS`,
`FINETUNE_EPOCHS`, `BATCH`, `PATIENCE`, `DEVICE`, `CACHE`, `RUN_NAME`) overrides
its key. `PRETRAIN_SOURCES` / `FINETUNE_SOURCE` name the datasets. See
`docs/MODEL_NOTES.md` for model + resolution guidance and
`docs/ADDING_CAMERA_FRAMES.md` for the dataset-improvement loop.

## Local dry run (no GPU)

```
pip install ultralytics onnxruntime pyyaml
# put a handful of images+labels in datasets/pool/train/{images,labels}
python scripts/make_val_split.py --base datasets/pool
MODEL=yolo11n.pt EPOCHS=1 IMGSZ=320 python scripts/train.py
# optional: add datasets/finetune/train/{images,labels} + a data/finetune.yaml
#           (copy data/pool.yaml, repoint `path`) to exercise phase 2
```
Confirms the wrapper, config parsing, and ONNX export path before spending GPU
time.

## Layout

```
train_config.yaml            defaults, env-overridable
data/pool.yaml               phase-1 data config (written by fetch_dataset.py)
data/finetune.yaml           phase-2 data config (written when FINETUNE_SOURCE set)
data/bullet-hole.yaml        legacy alias for pool.yaml (placeholder in git)
scripts/provision.sh         Vast on-start entrypoint
scripts/sources.py           source-spec parsing + download/normalize/merge helpers
scripts/fetch_dataset.py     build datasets/pool/ and datasets/finetune/
scripts/make_val_split.py    deterministic 85/15 split for a --base dir
scripts/train.py             two-phase ultralytics train + ONNX export
scripts/upload_artifacts.py  push weights/onnx/plots to a private HF model repo
Template/                    raw Vast launch config (hand-edited, source of truth)
docs/                        model notes + dataset loop
```
