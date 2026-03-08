from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _poly_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    acc = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        acc += x1 * y2 - x2 * y1
    return abs(acc) * 0.5


def _class_area_from_label(label_path: Path) -> dict[int, float]:
    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    out: dict[int, float] = {}
    if not text:
        return out
    for row in text.splitlines():
        parts = row.strip().split()
        if len(parts) < 7:
            continue
        try:
            cid = int(float(parts[0]))
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            continue
        if len(coords) < 6 or (len(coords) % 2) != 0:
            continue
        pts: list[tuple[float, float]] = []
        for i in range(0, len(coords), 2):
            x = min(1.0, max(0.0, coords[i]))
            y = min(1.0, max(0.0, coords[i + 1]))
            pts.append((x, y))
        area = _poly_area(pts)
        if area <= 0:
            continue
        out[cid] = out.get(cid, 0.0) + area
    return out


def _copy_pair(src_img: Path, src_lbl: Path, dst_img: Path, dst_lbl: Path) -> None:
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_img, dst_img)
    shutil.copy2(src_lbl, dst_lbl)


def _write_yaml(out_root: Path) -> Path:
    payload = {
        "path": str(out_root.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "root", 1: "stem", 2: "leaves"},
    }
    yml = out_root / "dataset.yaml"
    yml.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return yml


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ChronoRoot-focused dataset from a mixed YOLO-seg set.")
    parser.add_argument("--source", default="data/hf_multisource_mega10_fast", type=str)
    parser.add_argument("--out", default="data/hf_chrono_focus_v1", type=str)
    parser.add_argument("--train-token", default="__chrono_", type=str)
    parser.add_argument("--val-mode", choices=["full", "chrono"], default="full")
    parser.add_argument("--min-root-area", default=0.00004, type=float)
    parser.add_argument("--min-stem-area", default=0.00004, type=float)
    args = parser.parse_args()

    src = Path(args.source)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    copied = {"train": 0, "val": 0}
    class_counts = {"train": Counter(), "val": Counter()}

    src_train_lbl = src / "labels" / "train"
    src_train_img = src / "images" / "train"
    for lbl in sorted(src_train_lbl.glob("*.txt")):
        if args.train_token not in lbl.stem:
            continue
        areas = _class_area_from_label(lbl)
        if areas.get(0, 0.0) < float(args.min_root_area):
            continue
        if areas.get(1, 0.0) < float(args.min_stem_area):
            continue
        img = _find_image(src_train_img, lbl.stem)
        if img is None:
            continue
        dst_img = out / "images" / "train" / img.name
        dst_lbl = out / "labels" / "train" / lbl.name
        _copy_pair(img, lbl, dst_img, dst_lbl)
        copied["train"] += 1
        for cid in areas.keys():
            class_counts["train"][cid] += 1

    src_val_lbl = src / "labels" / "val"
    src_val_img = src / "images" / "val"
    for lbl in sorted(src_val_lbl.glob("*.txt")):
        if args.val_mode == "chrono" and args.train_token not in lbl.stem:
            continue
        img = _find_image(src_val_img, lbl.stem)
        if img is None:
            continue
        dst_img = out / "images" / "val" / img.name
        dst_lbl = out / "labels" / "val" / lbl.name
        _copy_pair(img, lbl, dst_img, dst_lbl)
        copied["val"] += 1
        areas = _class_area_from_label(lbl)
        for cid in areas.keys():
            class_counts["val"][cid] += 1

    yml = _write_yaml(out)
    report = {
        "source": str(src.resolve()),
        "out": str(out.resolve()),
        "train_token": args.train_token,
        "val_mode": args.val_mode,
        "min_root_area": float(args.min_root_area),
        "min_stem_area": float(args.min_stem_area),
        "copied": copied,
        "class_images_train": {str(k): int(v) for k, v in class_counts["train"].items()},
        "class_images_val": {str(k): int(v) for k, v in class_counts["val"].items()},
        "dataset_yaml": str(yml.resolve()),
    }
    rpt = out / "build_report.json"
    rpt.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
