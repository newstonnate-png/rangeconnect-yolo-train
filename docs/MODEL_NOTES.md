# Model choice & resolution notes

## Which YOLO

| Model | COCO mAP50-95 | CPU-ONNX @640 | Params | Notes |
|---|---|---|---|---|
| yolo11n | 39.5 | ~56 ms | 2.6 M | fastest; use if CPU latency is tight |
| **yolo11s** | **47.0** | ~90 ms | 9.4 M | **default** — best accuracy/effort for small holes on a CPU-only scoring PC |
| yolo11m | 51.5 | ~183 ms | 20 M | only if the scoring PC has real headroom |
| yolo26n | 40.9 | ~39 ms | 2.4 M | newest line; NMS-free; ProgLoss + STAL specifically target small objects; ~31–43 % faster CPU |
| yolo26s | 48.6 | ~87 ms | 9.5 M | same accuracy as 11s, faster; validate before trusting (new) |

CPU-ONNX figures are Ultralytics' reference numbers; the admin PC will be
slower, but with `FRAME_SKIP` in `scoring_app.py` and a static target the
effective rate only needs to be 2–4 FPS.

**Plan:** train `yolo11s` first (battle-tested, universal export support), then
run a second job with `MODEL=yolo26n.pt` and compare `mAP50` + real-frame CPU
latency. Pick the winner for `scoring_app.py`.

## Two-phase training

`scripts/train.py` runs phase 1 on `datasets/pool/` (pooled public data — a
generic hole-detection prior) then, if `data/finetune.yaml` exists, phase 2 on
`datasets/finetune/` (real zoom-lens frames on the printed target): fewer epochs
(`FINETUNE_EPOCHS`, default 40), `lr0=0.002`, `freeze=10` so the backbone holds
and only the head adapts to this camera. The first run has no fine-tune set and
phase 2 is skipped — that baseline model is still fine for wiring up
`scoring_app.py` while the real frames are collected. See
`ADDING_CAMERA_FRAMES.md`.

## Input resolution

Bullet holes are small; **resolution matters more than model size**. Defaults to
`IMGSZ=960`. If holes are tiny in the cropped zoom-lens frame and the GPU has
VRAM, try `IMGSZ=1280`. Keep training and inference `imgsz` equal — the ONNX
export bakes the size in (`dynamic=False`).

## Export

`train.py` exports `best.onnx` with `opset=12`, `simplify=True`, `dynamic=False`
— the widest-compatibility ONNX for `onnxruntime` CPU on the scoring PC. NCNN is
a later option if the range ever moves to an ARM board.
