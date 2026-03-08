from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


CLASS_ROOT = 0
CLASS_STEM = 1
CLASS_LEAVES = 2


@dataclass
class SampleMeta:
    path: Path
    class_area_norm: dict[int, float]


def polygon_area_norm(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def parse_label(label_path: Path) -> SampleMeta | None:
    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return None

    class_area: dict[int, float] = {}
    for row in text.splitlines():
        parts = row.strip().split()
        if len(parts) < 7:
            continue
        try:
            cid = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if len(vals) < 6 or (len(vals) % 2) != 0:
            continue
        pts = []
        for i in range(0, len(vals), 2):
            x = max(0.0, min(1.0, vals[i]))
            y = max(0.0, min(1.0, vals[i + 1]))
            pts.append((x, y))
        area = polygon_area_norm(pts)
        if area <= 0.0:
            continue
        class_area[cid] = class_area.get(cid, 0.0) + area

    if not class_area:
        return None
    return SampleMeta(path=label_path, class_area_norm=class_area)


def find_image_for_label(images_dir: Path, label_path: Path) -> Path | None:
    stem = label_path.stem
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        cand = images_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def copy_pair(src_img: Path, src_lbl: Path, out_img: Path, out_lbl: Path) -> None:
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_lbl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_img, out_img)
    shutil.copy2(src_lbl, out_lbl)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build root/stem boosted YOLO-seg dataset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/hf_multisource_rootstem_boost")
    parser.add_argument("--mega10", type=str, default="data/hf_multisource_mega10")
    parser.add_argument("--pack", type=str, default="data/roboflow_upload/plant_2f4ay_pack")
    parser.add_argument("--gold", type=str, default="data/roboflow_upload/plant_2f4ay_gold_pack")
    parser.add_argument("--max-mega10-train", type=int, default=1800)
    parser.add_argument("--max-mega10-val", type=int, default=300)
    parser.add_argument("--min-root-area-norm", type=float, default=0.00008)
    parser.add_argument("--min-stem-area-norm", type=float, default=0.00008)
    parser.add_argument("--oversample-rs-factor", type=int, default=2)
    args = parser.parse_args()

    random.seed(args.seed)

    out_root = Path(args.out)
    if out_root.exists():
        shutil.rmtree(out_root)
    (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    copied = {"train": 0, "val": 0}
    copied_by_source = {
        "mega10_train": 0,
        "mega10_val": 0,
        "pack_train": 0,
        "pack_val": 0,
        "gold_train": 0,
        "gold_val": 0,
        "oversample_train": 0,
    }

    def add_from_source(root: Path, source_name: str, split: str, sample_cap: int | None = None) -> list[tuple[Path, Path]]:
        pairs: list[tuple[Path, Path]] = []
        labels_dir = root / "labels" / split
        images_dir = root / "images" / split
        if not labels_dir.exists() or not images_dir.exists():
            return pairs

        for label_file in sorted(labels_dir.glob("*.txt")):
            image_file = find_image_for_label(images_dir, label_file)
            if image_file is None:
                continue
            pairs.append((image_file, label_file))

        if sample_cap is not None and len(pairs) > sample_cap:
            pairs = random.sample(pairs, sample_cap)

        for idx, (img, lbl) in enumerate(pairs):
            prefix = f"{source_name}_{split}_{idx:06d}"
            out_img = out_root / "images" / split / f"{prefix}{img.suffix.lower()}"
            out_lbl = out_root / "labels" / split / f"{prefix}.txt"
            copy_pair(img, lbl, out_img, out_lbl)
            copied[split] += 1
            copied_by_source[f"{source_name}_{split}"] += 1

        return pairs

    def filtered_mega10_pairs(split: str, cap: int) -> list[tuple[Path, Path, dict[int, float]]]:
        root = Path(args.mega10)
        labels_dir = root / "labels" / split
        images_dir = root / "images" / split
        out: list[tuple[Path, Path, dict[int, float]]] = []
        if not labels_dir.exists() or not images_dir.exists():
            return out

        for label_file in sorted(labels_dir.glob("*.txt")):
            meta = parse_label(label_file)
            if meta is None:
                continue
            root_area = meta.class_area_norm.get(CLASS_ROOT, 0.0)
            stem_area = meta.class_area_norm.get(CLASS_STEM, 0.0)
            if root_area < args.min_root_area_norm or stem_area < args.min_stem_area_norm:
                continue
            img = find_image_for_label(images_dir, label_file)
            if img is None:
                continue
            out.append((img, label_file, meta.class_area_norm))

        if len(out) > cap:
            out = random.sample(out, cap)
        return out

    mega_train = filtered_mega10_pairs("train", args.max_mega10_train)
    mega_val = filtered_mega10_pairs("val", args.max_mega10_val)

    for split, rows in (("train", mega_train), ("val", mega_val)):
        for idx, (img, lbl, _) in enumerate(rows):
            prefix = f"mega10f_{split}_{idx:06d}"
            out_img = out_root / "images" / split / f"{prefix}{img.suffix.lower()}"
            out_lbl = out_root / "labels" / split / f"{prefix}.txt"
            copy_pair(img, lbl, out_img, out_lbl)
            copied[split] += 1
            copied_by_source[f"mega10_{split}"] += 1

    pack_root = Path(args.pack)
    gold_root = Path(args.gold)
    add_from_source(pack_root, "pack", "train")
    add_from_source(pack_root, "pack", "val")
    add_from_source(gold_root, "gold", "train")
    add_from_source(gold_root, "gold", "val")

    if args.oversample_rs_factor > 1 and mega_train:
        rich_rows = sorted(
            mega_train,
            key=lambda item: item[2].get(CLASS_ROOT, 0.0) + item[2].get(CLASS_STEM, 0.0),
            reverse=True,
        )
        top_k = min(500, len(rich_rows))
        rich_rows = rich_rows[:top_k]
        for rep in range(args.oversample_rs_factor - 1):
            for idx, (img, lbl, _) in enumerate(rich_rows):
                prefix = f"mega10rs_r{rep+1}_{idx:06d}"
                out_img = out_root / "images" / "train" / f"{prefix}{img.suffix.lower()}"
                out_lbl = out_root / "labels" / "train" / f"{prefix}.txt"
                copy_pair(img, lbl, out_img, out_lbl)
                copied["train"] += 1
                copied_by_source["oversample_train"] += 1

    dataset_yaml = out_root / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {out_root.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: root",
                "  1: stem",
                "  2: leaves",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = {
        "seed": args.seed,
        "mega10_filtered_train": len(mega_train),
        "mega10_filtered_val": len(mega_val),
        "copied": copied,
        "copied_by_source": copied_by_source,
        "dataset_yaml": str(dataset_yaml),
    }
    report_path = out_root / "build_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
