Raw Vast.ai launch config for the RangeConnect YOLO training template. Hand-edit
this file only; apply it to the Vast console yourself. Tokens below are
placeholders — fill them in the Vast console, never commit real secrets.

## Instance

- **Image:** `ultralytics/ultralytics:latest`
- **Launch mode:** docker ENTRYPOINT
- **Disk:** 40 GB
- **GPU:** RTX 3090 or 5090, 1x

## On-start script (Vast "On-start Script" field)

```
git clone https://github.com/REPLACE_USER/rangeconnect-yolo-train /workspace/t && bash /workspace/t/scripts/provision.sh
```

## Docker options / env one-liner

```
-p 8888:8888 -e OPEN_BUTTON_PORT=8888 -e DATASET_SOURCE=roboflow -e ROBOFLOW_API_KEY=REPLACE_ME -e ROBOFLOW_WORKSPACE=justines-workspace-ls3un -e ROBOFLOW_PROJECT=justines-workspace-ls3un -e ROBOFLOW_VERSION=1 -e HF_TOKEN=REPLACE_ME -e HF_MODEL_REPO=REPLACE_USER/rc-bullet-hole-yolo -e MODEL=yolo11s.pt -e IMGSZ=960 -e EPOCHS=150 -e BATCH=16 -e PATIENCE=40 -e CACHE=ram -e ENABLE_JUPYTER=true -e JUPYTER_TOKEN=REPLACE_ME
```

## To benchmark YOLO26n instead

Change `-e MODEL=yolo26n.pt` (optionally `-e IMGSZ=640`) and re-launch. Artifacts
land in a separate `<stamp>-yolo26n` folder in the HF model repo.

## Offer search

```
vastai search offers 'gpu_name in [RTX_3090,RTX_5090] num_gpus=1 verified=true rentable=true direct_port_count>=1' -o dph
```
