from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


ALLOWED_IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


@dataclass
class SampleMeta:
    split: str
    image_path: Path
    label_path: Path
    blur_score: float
    brightness: float
    total_area_ratio: float
    class_instances: Counter
    class_area_px: Counter


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    if not isinstance(payload, dict):
        raise ValueError(f'Invalid yaml payload: {path}')
    return payload


def _resolve_dataset_root(dataset_yaml: Path) -> Path:
    payload = _load_yaml(dataset_yaml)
    root = Path(str(payload.get('path', dataset_yaml.parent)))
    if not root.is_absolute():
        root = (dataset_yaml.parent / root).resolve()
    return root


def _read_polygons(label_path: Path, image_w: int, image_h: int) -> tuple[Counter, Counter]:
    class_instances: Counter = Counter()
    class_area_px: Counter = Counter()
    if not label_path.exists():
        return class_instances, class_area_px

    text = label_path.read_text(encoding='utf-8', errors='ignore').strip()
    if not text:
        return class_instances, class_area_px

    for row in text.splitlines():
        parts = row.strip().split()
        if len(parts) < 7:
            continue
        try:
            class_id = int(float(parts[0]))
            coords = np.asarray([float(x) for x in parts[1:]], dtype=np.float32)
        except ValueError:
            continue
        if coords.size < 6 or coords.size % 2 != 0:
            continue

        pts = coords.reshape(-1, 2).copy()
        pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0) * float(image_w)
        pts[:, 1] = np.clip(pts[:, 1], 0.0, 1.0) * float(image_h)
        area = float(cv2.contourArea(pts.astype(np.float32)))
        if area <= 1.0:
            continue

        class_instances[class_id] += 1
        class_area_px[class_id] += area
    return class_instances, class_area_px


def _iter_samples(dataset_root: Path) -> list[SampleMeta]:
    samples: list[SampleMeta] = []
    for split in ('train', 'val'):
        img_dir = dataset_root / 'images' / split
        lbl_dir = dataset_root / 'labels' / split
        if not img_dir.exists():
            continue
        for img_path in img_dir.rglob('*'):
            if not img_path.is_file() or img_path.suffix.lower() not in ALLOWED_IMAGE_EXTS:
                continue
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image is None:
                continue

            h, w = image.shape[:2]
            if h <= 0 or w <= 0:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            brightness = float(gray.mean())

            lbl_path = lbl_dir / f'{img_path.stem}.txt'
            class_instances, class_area_px = _read_polygons(lbl_path, image_w=w, image_h=h)
            total_area = float(sum(class_area_px.values()))
            area_ratio = total_area / float(max(1, w * h))
            samples.append(
                SampleMeta(
                    split=split,
                    image_path=img_path,
                    label_path=lbl_path,
                    blur_score=blur,
                    brightness=brightness,
                    total_area_ratio=area_ratio,
                    class_instances=class_instances,
                    class_area_px=class_area_px,
                )
            )
    return samples


def _filter_samples(
    samples: list[SampleMeta],
    min_blur: float,
    min_brightness: float,
    max_brightness: float,
    min_area_ratio: float,
    max_area_ratio: float,
    min_instances_per_image: int,
) -> list[SampleMeta]:
    kept: list[SampleMeta] = []
    for s in samples:
        total_instances = int(sum(s.class_instances.values()))
        if total_instances < min_instances_per_image:
            continue
        if s.blur_score < min_blur:
            continue
        if s.brightness < min_brightness or s.brightness > max_brightness:
            continue
        if s.total_area_ratio < min_area_ratio or s.total_area_ratio > max_area_ratio:
            continue
        kept.append(s)
    return kept


def _aggregate_instances(samples: list[SampleMeta]) -> Counter:
    c: Counter = Counter()
    for s in samples:
        c.update(s.class_instances)
    return c


def _max_ratio(counter: Counter) -> float:
    non_zero = [int(v) for v in counter.values() if int(v) > 0]
    if not non_zero:
        return 0.0
    mn = max(1, min(non_zero))
    mx = max(non_zero)
    return float(mx) / float(mn)


def _balance_by_pruning(samples: list[SampleMeta], class_ratio_max: float, min_kept_images: int) -> list[SampleMeta]:
    kept = list(samples)
    if not kept:
        return kept

    # Greedy pruning of samples dominated by currently overrepresented classes.
    while len(kept) > max(1, min_kept_images):
        counts = _aggregate_instances(kept)
        non_zero = {k: int(v) for k, v in counts.items() if int(v) > 0}
        if not non_zero:
            break
        min_count = max(1, min(non_zero.values()))
        over_cls = [cid for cid, v in non_zero.items() if float(v) > float(min_count) * float(class_ratio_max)]
        if not over_cls:
            break

        # Choose one sample to drop: highest dominance of overrepresented classes,
        # and lowest support of rare classes.
        best_idx = -1
        best_score = -1.0
        for idx, s in enumerate(kept):
            total = float(sum(s.class_instances.values()))
            if total <= 0:
                continue
            over = float(sum(s.class_instances.get(cid, 0) for cid in over_cls))
            rare_support = float(
                sum(s.class_instances.get(cid, 0) for cid, v in non_zero.items() if v <= min_count * 1.1)
            )
            score = (over / total) - 0.35 * (rare_support / total)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx < 0:
            break
        del kept[best_idx]
    return kept


def _write_dataset_yaml(out_dir: Path, names_payload: Any) -> Path:
    payload = {
        'path': str(out_dir.resolve()).replace('\\', '/'),
        'train': 'images/train',
        'val': 'images/val',
        'names': names_payload,
    }
    target = out_dir / 'dataset.yaml'
    target.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding='utf-8')
    return target


def _copy_kept(kept: list[SampleMeta], out_dir: Path, source_root: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ('train', 'val'):
        (out_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    for s in kept:
        rel_img = s.image_path.relative_to(source_root / 'images' / s.split)
        dst_img = out_dir / 'images' / s.split / rel_img
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s.image_path, dst_img)

        dst_lbl = out_dir / 'labels' / s.split / f'{dst_img.stem}.txt'
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        if s.label_path.exists():
            shutil.copy2(s.label_path, dst_lbl)
        else:
            dst_lbl.write_text('', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Filter and rebalance YOLO-seg dataset by image quality and class skew.')
    parser.add_argument('--data-yaml', required=True, type=str)
    parser.add_argument('--out', default='data/hf_multisource_balanced', type=str)
    parser.add_argument('--min-blur', default=35.0, type=float)
    parser.add_argument('--min-brightness', default=30.0, type=float)
    parser.add_argument('--max-brightness', default=230.0, type=float)
    parser.add_argument('--min-area-ratio', default=0.0003, type=float)
    parser.add_argument('--max-area-ratio', default=0.92, type=float)
    parser.add_argument('--min-instances-per-image', default=1, type=int)
    parser.add_argument('--class-ratio-max', default=2.4, type=float)
    parser.add_argument('--min-kept-images', default=120, type=int)
    args = parser.parse_args()

    dataset_yaml = Path(args.data_yaml).resolve()
    source_payload = _load_yaml(dataset_yaml)
    names_payload = source_payload.get('names', {0: 'root', 1: 'stem', 2: 'leaves'})
    source_root = _resolve_dataset_root(dataset_yaml)
    all_samples = _iter_samples(source_root)

    filtered = _filter_samples(
        samples=all_samples,
        min_blur=float(args.min_blur),
        min_brightness=float(args.min_brightness),
        max_brightness=float(args.max_brightness),
        min_area_ratio=float(args.min_area_ratio),
        max_area_ratio=float(args.max_area_ratio),
        min_instances_per_image=int(args.min_instances_per_image),
    )
    balanced = _balance_by_pruning(
        samples=filtered,
        class_ratio_max=float(args.class_ratio_max),
        min_kept_images=int(args.min_kept_images),
    )

    out_dir = Path(args.out)
    _copy_kept(balanced, out_dir=out_dir, source_root=source_root)
    out_yaml = _write_dataset_yaml(out_dir=out_dir, names_payload=names_payload)

    before = _aggregate_instances(all_samples)
    after_filter = _aggregate_instances(filtered)
    after_balance = _aggregate_instances(balanced)
    report = {
        'source_data_yaml': str(dataset_yaml),
        'source_root': str(source_root),
        'output_data_yaml': str(out_yaml.resolve()),
        'images': {
            'all': len(all_samples),
            'after_filter': len(filtered),
            'after_balance': len(balanced),
            'train_after_balance': int(sum(1 for s in balanced if s.split == 'train')),
            'val_after_balance': int(sum(1 for s in balanced if s.split == 'val')),
        },
        'class_instances': {
            'before': {str(k): int(v) for k, v in before.items()},
            'after_filter': {str(k): int(v) for k, v in after_filter.items()},
            'after_balance': {str(k): int(v) for k, v in after_balance.items()},
        },
        'class_ratio': {
            'before_max_div_min': round(_max_ratio(before), 5),
            'after_filter_max_div_min': round(_max_ratio(after_filter), 5),
            'after_balance_max_div_min': round(_max_ratio(after_balance), 5),
            'target_max_ratio': float(args.class_ratio_max),
        },
        'filter_thresholds': {
            'min_blur': float(args.min_blur),
            'min_brightness': float(args.min_brightness),
            'max_brightness': float(args.max_brightness),
            'min_area_ratio': float(args.min_area_ratio),
            'max_area_ratio': float(args.max_area_ratio),
            'min_instances_per_image': int(args.min_instances_per_image),
            'min_kept_images': int(args.min_kept_images),
        },
    }

    report_path = out_dir / 'filter_balance_report.json'
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()

