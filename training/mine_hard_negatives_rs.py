from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image


CLASS_NAMES = {0: "root", 1: "stem", 2: "leaves"}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_gt_presence(label_path: Path) -> dict[int, int]:
    counts: dict[int, int] = Counter()
    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return {}
    for row in text.splitlines():
        parts = row.strip().split()
        if len(parts) < 3:
            continue
        try:
            cid = int(float(parts[0]))
        except ValueError:
            continue
        counts[cid] += 1
    return dict(counts)


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def collect_candidates(data_root: Path, split: str) -> list[tuple[Path, Path, dict[int, int]]]:
    labels_dir = data_root / "labels" / split
    images_dir = data_root / "images" / split
    out: list[tuple[Path, Path, dict[int, int]]] = []
    if not labels_dir.exists() or not images_dir.exists():
        return out
    for lbl in sorted(labels_dir.glob("*.txt")):
        gt = parse_gt_presence(lbl)
        if not gt:
            continue
        img = find_image(images_dir, lbl.stem)
        if img is None:
            continue
        out.append((img, lbl, gt))
    return out


def pred_presence(result, min_conf: float) -> dict[int, float]:
    best: dict[int, float] = {}
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return best
    cls_list = boxes.cls.tolist() if boxes.cls is not None else []
    conf_list = boxes.conf.tolist() if boxes.conf is not None else []
    for c_raw, conf_raw in zip(cls_list, conf_list):
        cid = int(c_raw)
        conf = float(conf_raw)
        if conf < min_conf:
            continue
        prev = best.get(cid, 0.0)
        if conf > prev:
            best[cid] = conf
    return best


def ensure_dirs(root: Path) -> None:
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "val").mkdir(parents=True, exist_ok=True)


def copy_pair(img: Path, lbl: Path, out_root: Path, split: str, name: str) -> None:
    out_img = out_root / "images" / split / f"{name}{img.suffix.lower()}"
    out_lbl = out_root / "labels" / split / f"{name}.txt"
    shutil.copy2(img, out_img)
    shutil.copy2(lbl, out_lbl)


def load_image_rgb(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"))
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine hard negatives for root/stem from labeled data.")
    parser.add_argument("--model", type=str, default="models/best_max.pt")
    parser.add_argument("--data-root", type=str, default="data/hf_multisource_mega10")
    parser.add_argument("--out", type=str, default="data/hard_mined_rootstem")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=1400)
    parser.add_argument("--val-limit", type=int, default=260)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--predict-conf", type=float, default=0.01)
    parser.add_argument("--presence-conf", type=float, default=0.10)
    parser.add_argument("--max-det", type=int, default=120)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=96)
    args = parser.parse_args()

    random.seed(args.seed)
    out_root = Path(args.out)
    if out_root.exists():
        shutil.rmtree(out_root)
    ensure_dirs(out_root)

    model = YOLO(args.model)

    all_reports: dict[str, dict] = {}
    hard_total = {"train": 0, "val": 0}
    hard_by_reason: Counter = Counter()

    for split, lim in (("train", args.train_limit), ("val", args.val_limit)):
        candidates = collect_candidates(Path(args.data_root), split)
        if len(candidates) > lim:
            candidates = random.sample(candidates, lim)
        preds = []
        for img, _, _ in candidates:
            arr = load_image_rgb(img)
            if arr is None:
                preds.append(None)
                continue
            try:
                sub_preds = model.predict(
                    source=arr,
                    conf=float(args.predict_conf),
                    imgsz=int(args.imgsz),
                    max_det=int(args.max_det),
                    device="cpu",
                    verbose=False,
                    stream=False,
                    batch=1,
                )
            except (cv2.error, MemoryError):
                preds.append(None)
                continue
            if sub_preds:
                preds.append(sub_preds[0])
            else:
                preds.append(None)

        split_rows = []
        for idx, pred in enumerate(preds):
            img, lbl, gt = candidates[idx]
            pp = pred_presence(pred, float(args.presence_conf)) if pred is not None else {}
            has_root_gt = gt.get(0, 0) > 0
            has_stem_gt = gt.get(1, 0) > 0
            has_root_pred = 0 in pp
            has_stem_pred = 1 in pp

            reasons = []
            if has_root_gt and not has_root_pred:
                reasons.append("miss_root")
            if has_stem_gt and not has_stem_pred:
                reasons.append("miss_stem")
            if not reasons:
                continue

            hard_name = f"hard_{split}_{hard_total[split]:06d}"
            copy_pair(img, lbl, out_root, split, hard_name)
            hard_total[split] += 1
            for r in reasons:
                hard_by_reason[r] += 1
            split_rows.append(
                {
                    "name": hard_name,
                    "image": str(img),
                    "label": str(lbl),
                    "reasons": reasons,
                    "gt": {CLASS_NAMES.get(k, str(k)): v for k, v in gt.items()},
                    "pred_conf": {CLASS_NAMES.get(k, str(k)): v for k, v in pp.items()},
                }
            )

        all_reports[split] = {
            "checked": len(candidates),
            "hard_found": len(split_rows),
            "rows": split_rows[:400],
        }

    ds_yaml = out_root / "dataset.yaml"
    ds_yaml.write_text(
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
        "model": args.model,
        "data_root": args.data_root,
        "hard_total": hard_total,
        "hard_by_reason": dict(hard_by_reason),
        "dataset_yaml": str(ds_yaml),
        "details": all_reports,
    }
    report_path = out_root / "mining_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
