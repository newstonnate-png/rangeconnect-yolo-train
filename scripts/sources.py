#!/usr/bin/env python3
"""Dataset source specs and helpers shared by fetch_dataset.py.

A *source spec* is a short string naming where one YOLO dataset comes from:

    roboflow:<workspace>/<project>/<version>   Roboflow (export is free-tier)
    url:<https url to a .zip or .tar.gz>        any hosted archive
    hf:<repo_id>                                private HF *dataset* repo (HF_TOKEN)

download_source() lands the raw export in a fresh directory; normalize_layout()
rewrites it to the predictable shape

    <dir>/{train,valid,test}/{images,labels}

so callers can merge several together without caring which exporter produced them.
"""
from __future__ import annotations

import os
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(d: Path) -> int:
    if not d.exists():
        return 0
    return sum(1 for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS)


def parse_spec(spec: str) -> tuple[str, str]:
    """'roboflow:a/b/3' -> ('roboflow', 'a/b/3'). Scheme defaults to roboflow."""
    spec = spec.strip()
    if ":" not in spec:
        return "roboflow", spec
    scheme, rest = spec.split(":", 1)
    scheme = scheme.lower()
    if scheme not in ("roboflow", "url", "hf"):
        # e.g. a bare https url without our 'url:' prefix
        if spec.startswith("http"):
            return "url", spec
        raise ValueError(f"unknown source scheme in {spec!r}")
    return scheme, rest


def download_source(spec: str, dest: Path) -> Path:
    scheme, rest = parse_spec(spec)
    dest.mkdir(parents=True, exist_ok=True)
    if scheme == "roboflow":
        return _download_roboflow(rest, dest)
    if scheme == "url":
        return _download_archive(rest, dest)
    if scheme == "hf":
        return _download_hf(rest, dest)
    raise ValueError(scheme)


def _download_roboflow(rest: str, dest: Path) -> Path:
    from roboflow import Roboflow

    parts = rest.split("/")
    if len(parts) != 3:
        raise ValueError(
            f"roboflow spec must be workspace/project/version, got {rest!r}"
        )
    workspace, project, version = parts[0], parts[1], int(parts[2])
    api_key = os.environ["ROBOFLOW_API_KEY"]
    fmt = os.environ.get("ROBOFLOW_FORMAT", "yolov8")
    print(f"  roboflow {workspace}/{project} v{version} as {fmt}")
    rf = Roboflow(api_key=api_key)
    ds = rf.workspace(workspace).project(project).version(version).download(
        fmt, location=str(dest)
    )
    return Path(ds.location)


def _download_archive(url: str, dest: Path) -> Path:
    name = url.split("?")[0].rstrip("/").split("/")[-1] or "archive"
    archive = dest / name
    print(f"  url {url}")
    with urllib.request.urlopen(url, timeout=120) as r, open(archive, "wb") as f:
        shutil.copyfileobj(r, f)
    if name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive) as t:
            t.extractall(dest)
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        raise ValueError(f"don't know how to unpack {name!r}")
    archive.unlink(missing_ok=True)
    return dest


def _download_hf(repo_id: str, dest: Path) -> Path:
    from huggingface_hub import snapshot_download

    print(f"  hf dataset {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
        local_dir=str(dest),
    )
    return dest


def normalize_layout(root: Path) -> None:
    """Rewrite an arbitrary YOLO export under `root` to
    root/{train,valid,test}/{images,labels}.

    Handles: a single nested export folder; 'val' vs 'valid'; a flat
    images/labels pair with no split (put it all in train/).
    """
    # unwrap one level of nesting: root/<x>/train/...
    if not (root / "train").exists() and not (root / "images").exists():
        for child in sorted(p for p in root.iterdir() if p.is_dir()):
            if (child / "train").exists() or (child / "images").exists():
                for sub in child.iterdir():
                    target = root / sub.name
                    if not target.exists():
                        shutil.move(str(sub), str(target))
                shutil.rmtree(child, ignore_errors=True)
                break

    if (root / "val").exists() and not (root / "valid").exists():
        (root / "val").rename(root / "valid")

    # flat images/labels with no split -> train/
    if (root / "images").exists() and not (root / "train").exists():
        (root / "train").mkdir(exist_ok=True)
        for sub in ("images", "labels"):
            if (root / sub).exists():
                shutil.move(str(root / sub), str(root / "train" / sub))

    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)


def _read_class_names(root: Path) -> list[str]:
    """Pull the ordered class-name list from a YOLO export's data.yaml.

    Handles `names: [a, b]` and `names: {0: a, 1: b}`. Returns [] when there is
    no data.yaml or no names key — callers then treat every class as keep+collapse.
    """
    try:
        import yaml
    except ImportError:  # pragma: no cover - yaml ships with ultralytics
        return []
    for cand in ("data.yaml", "data.yml"):
        p = root / cand
        if not p.exists():
            continue
        doc = yaml.safe_load(p.read_text()) or {}
        names = doc.get("names")
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
        if isinstance(names, list):
            return [str(n) for n in names]
    return []


def collapse_labels(root: Path, drop_name_substrings: "list[str] | None" = None) -> tuple[int, int]:
    """Rewrite every YOLO label under root/{train,valid,test}/labels to a single
    class 0, so a multi-class export can join a single-class ('bullet-hole') pool.

    Boxes whose class *name* (looked up in the export's data.yaml `names`)
    contains any string in `drop_name_substrings` (case-insensitive) are removed
    entirely — e.g. 'Target', 'black_contour', 'sticker'. Every surviving box is
    remapped to class id 0.

    Returns (label_files_rewritten, boxes_dropped).
    """
    drops = [s.strip().lower() for s in (drop_name_substrings or []) if s.strip()]
    names = _read_class_names(root)
    drop_ids = {
        i for i, n in enumerate(names)
        if any(s in n.lower() for s in drops)
    }
    if names and drop_ids:
        print(
            f"  collapse: dropping classes "
            f"{sorted(names[i] for i in drop_ids)}; remapping the rest -> 0"
        )
    else:
        print("  collapse: remapping all classes -> 0")

    rewritten = dropped = 0
    for split in ("train", "valid", "test"):
        lbl_dir = root / split / "labels"
        if not lbl_dir.exists():
            continue
        for txt in lbl_dir.glob("*.txt"):
            out_lines = []
            for line in txt.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    cid = int(float(parts[0]))
                except (ValueError, IndexError):
                    continue
                if cid in drop_ids:
                    dropped += 1
                    continue
                parts[0] = "0"
                out_lines.append(" ".join(parts))
            txt.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
            rewritten += 1
    return rewritten, dropped


def merge_into(src_root: Path, dst_root: Path, prefix: str) -> int:
    """Copy every image+label pair from src_root/{split} into dst_root/{split},
    renaming to `<prefix>__<original>` so pools from different sources never
    collide. Returns the number of images copied.
    """
    copied = 0
    for split in ("train", "valid", "test"):
        s_img, s_lbl = src_root / split / "images", src_root / split / "labels"
        d_img, d_lbl = dst_root / split / "images", dst_root / split / "labels"
        d_img.mkdir(parents=True, exist_ok=True)
        d_lbl.mkdir(parents=True, exist_ok=True)
        if not s_img.exists():
            continue
        for img in s_img.iterdir():
            if img.suffix.lower() not in IMG_EXTS:
                continue
            shutil.copy2(img, d_img / f"{prefix}__{img.name}")
            lbl = s_lbl / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, d_lbl / f"{prefix}__{lbl.name}")
            copied += 1
    return copied


def write_data_yaml(path: Path, dataset_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Auto-generated by scripts/fetch_dataset.py -- do not edit by hand.\n"
        f"path: {dataset_root.as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 1\n"
        "names: [bullet-hole]\n"
    )
