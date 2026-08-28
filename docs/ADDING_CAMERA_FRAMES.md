# Adding real range-camera frames (the phase-2 fine-tune set)

The seed dataset (`justines-workspace-ls3un` on Roboflow, 1290 images, class
`bullet-hole`) was labelled on phone-camera imagery. The biggest accuracy lever
is fine-tuning on frames from the **actual mounted camera + printed target** —
this is `datasets/finetune/`, phase 2 of `scripts/train.py`.

## Loop

1. **Capture frames.** Run `py GunRangeApp3/camera_view.py`, aim at the printed
   target, and use the capture keys (single `S`, or timed burst) — it saves the
   frame **already cropped to the zoom lens and rotated**, i.e. exactly what the
   detector scores. Aim for 200–400 frames spanning:
   - low / medium / high hole counts
   - the lane's real lighting, plus any glare or shadow that actually occurs
   - a few holes deliberately on / near zone boundaries (these also seed the
     fuzzy scorer's test fixtures)
   - a few clean full-target frames with no holes

2. **Upload to Roboflow.** Same workspace/project as the seed set. Drag the
   frames in.

3. **Label.** One class, `bullet-hole`. Box each hole tightly — centroid
   precision is what drives correct zone scoring.

4. **Generate a new version.** Minimal preprocessing (auto-orient + resize-to-fit
   is fine). Leave augmentation to `train.py`. Note the new **version number**.

5. **Retrain.** Re-launch the Vast instance, adding
   `-e FINETUNE_SOURCE=roboflow:<workspace>/<project>/<n>` (keep
   `PRETRAIN_SOURCES` as-is). Phase 1 pretrains on the pool; phase 2 fine-tunes
   on the new frames. Artifacts land in a fresh `<stamp>-<model>` folder in the
   HF model repo, so older runs are kept.

6. **Compare.** Check `results.png` / `mAP50` for the fine-tune run, then
   CPU-test the new `best.onnx` on a held-out real frame.

## After the first model exists

`scripts/prelabel.py` (added once there is a trained ONNX) runs the model over a
folder of freshly captured frames and writes YOLO `.txt` pre-labels, so round 2+
labelling in Roboflow is review-and-correct instead of from scratch.

## Datasets NOT used

- `lonlonago` "Target on Bullet Hole" 9856-image set — **$89 Stripe paywall**;
  only preview images are public.
- MDPI "YOLOv8 + Detectron2 shooting cards" set — **flatbed scans of concentric
  paper cards**. Wrong domain for a live IP-camera view of a colour silhouette.
  The paper's scoring method is worth reading; the images are not worth training
  on.

## Why Roboflow versioning

`FINETUNE_SOURCE=roboflow:.../<n>` is a complete, reproducible record of what a
given `best.onnx` was trained on. Dataset export via the API is free — only
hosted inference and Roboflow Train cost money, and this template uses neither.
