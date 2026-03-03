from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


CLASS_NAMES = {0: "root", 1: "stem", 2: "leaves"}
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]


@dataclass
class ImageRecord:
    image_path: Path
    label_path: Path
    rel_stem: str
    classes: set[int]
    score: float


def find_image_for_label(labels_root: Path, images_root: Path, label_path: Path) -> Path | None:
    rel = label_path.relative_to(labels_root)
    stem = rel.with_suffix("")
    for ext in IMAGE_EXTS:
        candidate = images_root / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def parse_classes(label_path: Path) -> set[int]:
    classes: set[int] = set()
    try:
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            cls = int(float(parts[0]))
            if cls in CLASS_NAMES:
                classes.add(cls)
    except Exception:
        return set()
    return classes


def quality_score(image_path: Path, classes: set[int]) -> float:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return -1e9
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if max(h, w) > 900:
        scale = 900.0 / float(max(h, w))
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())

    dark_ratio = float((gray < 20).mean())
    bright_ratio = float((gray > 235).mean())
    exposure_penalty = max(0.0, dark_ratio - 0.35) + max(0.0, bright_ratio - 0.35)

    sharp_term = min(1.0, sharpness / 350.0)
    contrast_term = min(1.0, contrast / 55.0)

    class_bonus = 0.0
    if classes == {0, 1, 2}:
        class_bonus += 0.55
    elif {0, 1}.issubset(classes):
        class_bonus += 0.35
    elif {1, 2}.issubset(classes):
        class_bonus += 0.25
    elif 0 in classes:
        class_bonus += 0.15

    return 0.55 * sharp_term + 0.40 * contrast_term + class_bonus - 0.35 * exposure_penalty


def build_records(dataset_root: Path) -> list[ImageRecord]:
    labels_root = dataset_root / "labels"
    images_root = dataset_root / "images"
    if not labels_root.exists() or not images_root.exists():
        raise FileNotFoundError("Dataset must contain 'images/' and 'labels/' directories.")

    label_files = sorted(labels_root.rglob("*.txt"))
    records: list[ImageRecord] = []
    for label_path in label_files:
        cls_set = parse_classes(label_path)
        if not cls_set:
            continue
        image_path = find_image_for_label(labels_root, images_root, label_path)
        if image_path is None:
            continue
        rel_stem = str(label_path.relative_to(labels_root).with_suffix("")).replace("\\", "/")
        score = quality_score(image_path, cls_set)
        if score < -1e8:
            continue
        records.append(
            ImageRecord(
                image_path=image_path,
                label_path=label_path,
                rel_stem=rel_stem,
                classes=cls_set,
                score=score,
            )
        )
    return records


def select_balanced(records: list[ImageRecord], max_images: int, seed: int) -> list[ImageRecord]:
    rnd = random.Random(seed)
    records_sorted = sorted(records, key=lambda r: r.score, reverse=True)

    target_per_class = max(1, int(max_images * 0.34))
    selected: dict[str, ImageRecord] = {}
    class_img_count = {0: 0, 1: 0, 2: 0}

    for cls in [0, 1, 2]:
        for rec in records_sorted:
            if len(selected) >= max_images:
                break
            if cls not in rec.classes:
                continue
            if class_img_count[cls] >= target_per_class:
                break
            if rec.rel_stem in selected:
                continue
            selected[rec.rel_stem] = rec
            for c in rec.classes:
                class_img_count[c] += 1

    for rec in records_sorted:
        if len(selected) >= max_images:
            break
        if rec.rel_stem in selected:
            continue
        selected[rec.rel_stem] = rec
        for c in rec.classes:
            class_img_count[c] += 1

    selected_list = list(selected.values())
    rnd.shuffle(selected_list)
    return selected_list[:max_images]


def copy_pack(selected: list[ImageRecord], out_dir: Path, val_ratio: float, seed: int) -> dict:
    rnd = random.Random(seed)
    out_images_train = out_dir / "images" / "train"
    out_images_val = out_dir / "images" / "val"
    out_labels_train = out_dir / "labels" / "train"
    out_labels_val = out_dir / "labels" / "val"
    for p in [out_images_train, out_images_val, out_labels_train, out_labels_val]:
        p.mkdir(parents=True, exist_ok=True)

    items = selected[:]
    rnd.shuffle(items)
    val_n = max(1, int(round(len(items) * val_ratio)))
    val_stems = {r.rel_stem for r in items[:val_n]}

    class_image_count = {0: 0, 1: 0, 2: 0}
    split_counts = {"train": 0, "val": 0}

    for rec in items:
        split = "val" if rec.rel_stem in val_stems else "train"
        img_dst_dir = out_images_val if split == "val" else out_images_train
        lbl_dst_dir = out_labels_val if split == "val" else out_labels_train

        img_dst = img_dst_dir / rec.image_path.name
        lbl_dst = lbl_dst_dir / rec.label_path.name
        shutil.copy2(rec.image_path, img_dst)
        shutil.copy2(rec.label_path, lbl_dst)

        split_counts[split] += 1
        for c in rec.classes:
            class_image_count[c] += 1

    dataset_yaml = {
        "path": str(out_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "root", 1: "stem", 2: "leaves"},
    }
    (out_dir / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")

    report = {
        "selected_images": len(items),
        "split_counts": split_counts,
        "class_image_count": {CLASS_NAMES[k]: v for k, v in class_image_count.items()},
    }
    (out_dir / "pack_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare high-quality YOLO-seg pack for Roboflow upload.")
    parser.add_argument("--source", default="data/hf_multisource_balanced_auto_v4_relaxed45")
    parser.add_argument("--out", default="data/roboflow_upload/plant_2f4ay_pack")
    parser.add_argument("--max-images", type=int, default=900)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    out_dir = Path(args.out)
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = build_records(source)
    if not records:
        raise RuntimeError("No valid image/label records found.")

    selected = select_balanced(records, max_images=max(10, args.max_images), seed=args.seed)
    report = copy_pack(selected, out_dir=out_dir, val_ratio=max(0.01, min(0.4, args.val_ratio)), seed=args.seed)

    print(json.dumps(
        {
            "source": str(source),
            "out": str(out_dir),
            "records_total": len(records),
            "selected": len(selected),
            "report": report,
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
