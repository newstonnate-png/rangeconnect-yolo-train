# Adding real range-camera frames to the dataset

The seed dataset (`justines-workspace-ls3un` on Roboflow, 1290 images, class
`bullet-hole`) was labelled on phone-camera imagery. The biggest accuracy lever
is fine-tuning on frames from the **actual mounted camera + printed target**.

## Loop

1. **Capture frames.** With a relay running, pull stills from the RTSP zoom lens
   (the lens `scoring_app.py` scores from). Aim for 100–200 frames spanning:
   - low / medium / high hole counts on the target
   - the lane's normal lighting, plus any glare or shadow that actually occurs
   - a few with the target partly occluded / at the edge of frame
   `GunRangeApp3/probe_camera.py` already grabs sample frames; or use
   `camera_view.py` and press `S`.

2. **Upload to Roboflow.** Same workspace/project as the seed set
   (`justines-workspace-ls3un`). Drag the frames in.

3. **Label.** One class, `bullet-hole`. Box each hole tightly. Roboflow's
   free tier covers annotation.

4. **Generate a new version.** Keep preprocessing minimal (auto-orient +
   resize-to-fit is fine). Augmentations: leave to `train.py` — it already does
   mosaic + flips. Note the new **version number**.

5. **Retrain.** Re-launch the Vast instance with the new
   `-e ROBOFLOW_VERSION=<n>`. Everything else is unchanged. Artifacts land in a
   fresh `<stamp>-<model>` folder in the HF model repo, so old runs are kept.

6. **Compare.** Check `results.png` / `mAP50` against the previous run, then
   CPU-test the new `best.onnx` on a held-out real frame.

## Why Roboflow (not a manual split)

Roboflow versions the dataset, so `ROBOFLOW_VERSION` is a complete, reproducible
record of what a given `best.onnx` was trained on. Dataset export via the API is
free — only hosted inference and Roboflow Train cost money, and this template
uses neither.
