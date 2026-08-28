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
git clone https://github.com/newstonnate-png/rangeconnect-yolo-train /workspace/t && bash /workspace/t/scripts/provision.sh
```

> Private repo — the Vast instance needs read access. Either make the repo public,
> or use a clone URL with a read-only PAT:
> `git clone https://<user>:<PAT>@github.com/newstonnate-png/rangeconnect-yolo-train ...`

## Docker options / env one-liner

Baseline run (pretrain pool only, no fine-tune set yet):

```
-p 8888:8888 -e OPEN_BUTTON_PORT=8888 -e ROBOFLOW_API_KEY=REPLACE_ME -e PRETRAIN_SOURCES=roboflow:justines-workspace-ls3un/justines-workspace-ls3un/1,roboflow:project-bat-bullet-hole-detection/bullet-hole-object-detection/1 -e FINETUNE_SOURCE= -e HF_TOKEN=REPLACE_ME -e HF_MODEL_REPO=NewstonNate13/rc-bullet-hole-yolo -e MODEL=yolo11s.pt -e IMGSZ=960 -e EPOCHS=150 -e FINETUNE_EPOCHS=40 -e BATCH=16 -e PATIENCE=40 -e CACHE=ram -e ENABLE_JUPYTER=true -e JUPYTER_TOKEN=REPLACE_ME
```

In-domain run (after camera frames are labelled as a new Roboflow version, e.g. v2):

```
... -e FINETUNE_SOURCE=roboflow:justines-workspace-ls3un/justines-workspace-ls3un/2 ...
```

## Notes

- `PRETRAIN_SOURCES` — comma list. Specs: `roboflow:<ws>/<proj>/<ver>` |
  `url:<zip/tar.gz>` | `hf:<dataset repo>`. Fill in the real version numbers
  from the Roboflow dataset pages before launching.
- To benchmark YOLO26n: `-e MODEL=yolo26n.pt` (optionally `-e IMGSZ=640`).
  Artifacts land in a separate `<stamp>-yolo26n` folder in the HF model repo.
- `FINETUNE_SOURCE=` (empty) skips phase 2 — the baseline path.
- Excluded on purpose: `lonlonago` 9856-image set ($89 paywall), MDPI
  "shooting cards" set (flatbed scans — wrong domain for a live IP camera).

## Offer search

```
vastai search offers 'gpu_name in [RTX_3090,RTX_5090] num_gpus=1 verified=true rentable=true direct_port_count>=1' -o dph
```
