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

Secrets live in `../.env` (gitignored) — copy the values in. Real tokens must be
pasted in the Vast console, never committed here.

Pretrain pool = four public CC-BY-4.0 Roboflow Universe datasets (~2,400 base
images), each folded down to the single `bullet-hole` class by
`fetch_dataset.py` (drops `Target` / `*contour*` boxes, remaps the rest to 0):

| spec | base img | native classes |
|---|---|---|
| `shootingscoreai/bullet-hole-webcam/4` | 885 | bullet-hole, black-contour |
| `project-bat-bullet-hole-detection/bullet-hole-object-detection/30` | 1243 | Bullet_0-10, Target, black_contour |
| `shootingscoreai/new-bullet-hole/5` | 165 | bullet_hole, black_contour |
| `jasons-workspace-og0qz/bullet-hole-detection-3wyec/24` | 123 | Bullet-Hole-detection (already 1) |

### Run A — yolo11s (primary), baseline / phase-1 only

```
-p 8888:8888 -e OPEN_BUTTON_PORT=8888 -e ROBOFLOW_API_KEY=REPLACE_ME -e PRETRAIN_SOURCES=roboflow:shootingscoreai/bullet-hole-webcam/4,roboflow:project-bat-bullet-hole-detection/bullet-hole-object-detection/30,roboflow:shootingscoreai/new-bullet-hole/5,roboflow:jasons-workspace-og0qz/bullet-hole-detection-3wyec/24 -e FINETUNE_SOURCE= -e HF_TOKEN=REPLACE_ME -e HF_MODEL_REPO=NewstonNate13/rc-bullet-hole-yolo -e MODEL=yolo11s.pt -e IMGSZ=960 -e EPOCHS=150 -e FINETUNE_EPOCHS=40 -e BATCH=16 -e PATIENCE=40 -e CACHE=ram -e ENABLE_JUPYTER=true -e JUPYTER_TOKEN=REPLACE_ME
```

### Run B — yolo26n (benchmark)

Same one-liner with `-e MODEL=yolo26n.pt -e IMGSZ=640`. Artifacts land in a
separate `<stamp>-yolo26n` folder in the HF model repo, so both runs coexist.

### In-domain run (later, once camera frames are labelled as a Roboflow version)

```
... -e FINETUNE_SOURCE=roboflow:<your-ws>/<your-proj>/<ver> ...
```

## Notes

- `PRETRAIN_SOURCES` — comma list. Specs: `roboflow:<ws>/<proj>/<ver>` |
  `url:<zip/tar.gz>` | `hf:<dataset repo>`.
- `POOL_DROP_CLASSES` (default `contour,target,sticker,shell,casing,hull`) —
  class-name fragments dropped during the single-class collapse. Override if a
  new source names its hole class something that matches one of these.
- `FINETUNE_SOURCE=` (empty) skips phase 2 — the baseline path.
- `justines-workspace-ls3un` (the old default) was an external account we don't
  control — dropped in favour of the public pool above.
- Also excluded: `lonlonago` 9856-image set ($89 paywall), MDPI "shooting cards"
  set (flatbed scans — wrong domain for a live IP camera).

## Offer search

```
vastai search offers 'gpu_name in [RTX_3090,RTX_5090] num_gpus=1 verified=true rentable=true direct_port_count>=1' -o dph
```
