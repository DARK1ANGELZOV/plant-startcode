from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


CLASS_NAMES = {0: "root", 1: "stem", 2: "leaves"}
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]


@dataclass
class InstanceGeom:
    class_id: int
    area_px: float
    area_ratio: float
    bbox_ar: float


@dataclass
class Record:
    image_path: Path
    label_path: Path
    stem: str
    classes: set[int]
    score: float


def _find_image(images_root: Path, labels_root: Path, label_path: Path) -> Path | None:
    rel = label_path.relative_to(labels_root).with_suffix("")
    for ext in IMAGE_EXTS:
        p = images_root / f"{rel}{ext}"
        if p.exists():
            return p
    return None


def _poly_area(points_xy: np.ndarray) -> float:
    # Shoelace formula
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _parse_label_geoms(label_path: Path, img_w: int, img_h: int) -> list[InstanceGeom] | None:
    geoms: list[InstanceGeom] = []
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7 or len(parts) % 2 == 0:
            return None
        try:
            cls = int(float(parts[0]))
            coords = np.asarray([float(v) for v in parts[1:]], dtype=np.float32).reshape(-1, 2)
        except Exception:
            return None
        if cls not in CLASS_NAMES or coords.shape[0] < 3:
            return None
        if (coords < 0).any() or (coords > 1).any():
            return None

        pts = coords.copy()
        pts[:, 0] *= float(img_w)
        pts[:, 1] *= float(img_h)

        area = _poly_area(pts)
        if area <= 0:
            continue

        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        bw = max(1e-6, float(x_max - x_min))
        bh = max(1e-6, float(y_max - y_min))
        ar = max(bw / bh, bh / bw)
        geoms.append(
            InstanceGeom(
                class_id=cls,
                area_px=area,
                area_ratio=float(area / max(1.0, float(img_w * img_h))),
                bbox_ar=float(ar),
            )
        )
    return geoms


def _source_prefix(stem: str) -> str:
    s = Path(stem).name.lower()
    if s.startswith("chrono_"):
        return "chrono"
    if s.startswith("plantorg_"):
        return "plantorg"
    return "other"


def _passes_geometry_rules(stem: str, geoms: list[InstanceGeom]) -> tuple[bool, str]:
    if not geoms:
        return False, "no_instances"

    total_area_ratio = float(sum(g.area_ratio for g in geoms))
    if total_area_ratio < 0.002 or total_area_ratio > 0.95:
        return False, "total_area_out_of_range"

    by_class = defaultdict(list)
    for g in geoms:
        by_class[g.class_id].append(g)

    # Remove extremely noisy annotations.
    tiny_count = sum(1 for g in geoms if g.area_ratio < 0.00003)
    if tiny_count > 20:
        return False, "too_many_tiny_instances"
    if len(geoms) > 140:
        return False, "too_many_instances"

    root_area = float(sum(g.area_ratio for g in by_class[0]))
    stem_area = float(sum(g.area_ratio for g in by_class[1]))
    leaf_area = float(sum(g.area_ratio for g in by_class[2]))

    if root_area > 0.55 or stem_area > 0.55 or leaf_area > 0.92:
        return False, "class_area_implausible"

    # Source-aware constraints.
    src = _source_prefix(stem)
    if src == "plantorg" and root_area > 0.0:
        return False, "plantorg_has_root"
    if src == "chrono" and leaf_area > 0.80:
        return False, "chrono_leaf_area_too_large"

    # Stem should not be only tiny dust when present.
    if by_class[1]:
        stem_max = max(g.area_ratio for g in by_class[1])
        if stem_max < 0.00012:
            return False, "stem_too_tiny"

    # Root should be moderately elongated when present.
    if by_class[0]:
        root_ar = max(g.bbox_ar for g in by_class[0])
        if root_ar < 1.2:
            return False, "root_not_elongated"

    return True, "ok"


def _image_quality_score(image_path: Path) -> tuple[float, bool, str]:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return -1e9, False, "image_load_failed"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    mean_brightness = float(gray.mean())

    if lap_var < 22.0:
        return -1e9, False, "too_blurry"
    if contrast < 16.0:
        return -1e9, False, "too_low_contrast"
    if mean_brightness < 22.0 or mean_brightness > 235.0:
        return -1e9, False, "bad_exposure"

    score = 0.60 * min(1.0, lap_var / 260.0) + 0.40 * min(1.0, contrast / 58.0)
    return float(score), True, "ok"


def _build_records(source: Path) -> tuple[list[Record], dict[str, int]]:
    labels_root = source / "labels"
    images_root = source / "images"
    label_files = sorted(labels_root.rglob("*.txt"))

    reject_stats: Counter[str] = Counter()
    records: list[Record] = []
    for lb in label_files:
        img = _find_image(images_root, labels_root, lb)
        if img is None:
            reject_stats["missing_image"] += 1
            continue
        im = cv2.imread(str(img), cv2.IMREAD_COLOR)
        if im is None:
            reject_stats["image_load_failed"] += 1
            continue
        h, w = im.shape[:2]

        geoms = _parse_label_geoms(lb, w, h)
        if geoms is None:
            reject_stats["label_parse_failed"] += 1
            continue

        stem = str(lb.relative_to(labels_root).with_suffix("")).replace("\\", "/")
        ok_geom, reason = _passes_geometry_rules(stem, geoms)
        if not ok_geom:
            reject_stats[reason] += 1
            continue

        q_score, ok_q, q_reason = _image_quality_score(img)
        if not ok_q:
            reject_stats[q_reason] += 1
            continue

        classes = {g.class_id for g in geoms}
        coverage_bonus = 0.0
        if classes == {0, 1, 2}:
            coverage_bonus += 0.55
        elif {0, 1}.issubset(classes):
            coverage_bonus += 0.30
        elif {1, 2}.issubset(classes):
            coverage_bonus += 0.20
        elif 0 in classes:
            coverage_bonus += 0.12

        score = q_score + coverage_bonus
        records.append(Record(image_path=img, label_path=lb, stem=stem, classes=classes, score=score))

    return records, dict(reject_stats)


def _select_balanced(records: list[Record], max_images: int, seed: int) -> list[Record]:
    rnd = random.Random(seed)
    recs = sorted(records, key=lambda r: r.score, reverse=True)

    # Hard quotas to ensure all classes represented well.
    q_root = int(max_images * 0.35)
    q_stem = int(max_images * 0.45)
    q_leaf = int(max_images * 0.45)

    sel: dict[str, Record] = {}
    cls_img_count = {0: 0, 1: 0, 2: 0}

    def take_if_needed(cls_id: int, quota: int) -> None:
        for r in recs:
            if len(sel) >= max_images:
                return
            if cls_img_count[cls_id] >= quota:
                return
            if cls_id not in r.classes:
                continue
            if r.stem in sel:
                continue
            sel[r.stem] = r
            for c in r.classes:
                cls_img_count[c] += 1

    take_if_needed(0, q_root)
    take_if_needed(1, q_stem)
    take_if_needed(2, q_leaf)

    for r in recs:
        if len(sel) >= max_images:
            break
        if r.stem in sel:
            continue
        sel[r.stem] = r
        for c in r.classes:
            cls_img_count[c] += 1

    out = list(sel.values())
    rnd.shuffle(out)
    return out[:max_images]


def _copy_pack(selected: list[Record], out: Path, val_ratio: float, seed: int) -> dict:
    rnd = random.Random(seed)
    if out.exists():
        shutil.rmtree(out)
    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "val").mkdir(parents=True, exist_ok=True)

    items = selected[:]
    rnd.shuffle(items)
    val_n = max(1, int(round(len(items) * val_ratio)))
    val_set = {r.stem for r in items[:val_n]}

    split_counts = {"train": 0, "val": 0}
    source_counts = Counter()
    class_img_counts = {0: 0, 1: 0, 2: 0}

    for r in items:
        split = "val" if r.stem in val_set else "train"
        shutil.copy2(r.image_path, out / "images" / split / r.image_path.name)
        shutil.copy2(r.label_path, out / "labels" / split / r.label_path.name)

        split_counts[split] += 1
        source_counts[_source_prefix(r.stem)] += 1
        for c in r.classes:
            class_img_counts[c] += 1

    dataset_yaml = {
        "path": str(out.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "root", 1: "stem", 2: "leaves"},
    }
    (out / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")

    report = {
        "selected_images": len(items),
        "split_counts": split_counts,
        "source_counts": dict(source_counts),
        "class_image_counts": {CLASS_NAMES[k]: v for k, v in class_img_counts.items()},
    }
    (out / "gold_pack_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare strict gold Roboflow pack.")
    parser.add_argument("--source", default="data/hf_multisource_balanced_auto_v4_relaxed45")
    parser.add_argument("--out", default="data/roboflow_upload/plant_2f4ay_gold_pack")
    parser.add_argument("--max-images", type=int, default=600)
    parser.add_argument("--val-ratio", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)

    records, rejects = _build_records(source)
    if not records:
        raise RuntimeError("No records passed strict filtering.")

    selected = _select_balanced(records, max_images=max(50, int(args.max_images)), seed=args.seed)
    report = _copy_pack(selected, out=out, val_ratio=max(0.05, min(0.3, args.val_ratio)), seed=args.seed)

    print(
        json.dumps(
            {
                "source": str(source),
                "out": str(out),
                "records_after_strict_filter": len(records),
                "selected": len(selected),
                "reject_reasons": rejects,
                "report": report,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
