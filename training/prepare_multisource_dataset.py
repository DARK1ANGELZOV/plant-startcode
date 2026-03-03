from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import nibabel as nib
import numpy as np
import yaml
import ijson

# Workaround for Windows environments where a compatible torch build is installed
# in a short custom site-packages path (e.g. C:\ptpkgs).
_extra_site = os.getenv('AGRO_EXTRA_SITE_PACKAGES', r'C:\ptpkgs').strip()
if _extra_site and os.path.isdir(_extra_site) and _extra_site not in sys.path:
    sys.path.insert(0, _extra_site)

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download


CLASS_NAMES = ['root', 'stem', 'leaves']

# ChronoRoot2 labels:
# 1 Main Root, 2 Lateral Roots, 3 Seed, 4 Hypocotyl, 5 Leaves, 6 Petiole
CHRONOROOT_TO_TARGET = {
    0: [1, 2],  # root
    1: [4, 6],  # stem-like tissues
    2: [5],     # leaves
}

# PlantOrgans labels:
# 0 void, 1 Fruit, 2 Leaf, 3 Flower, 4 Stem
PLANTORGANS_TO_TARGET = {
    1: [4],  # stem
    2: [2],  # leaves
}


@dataclass
class BuildStats:
    total_images: int = 0
    written_images: int = 0
    empty_masks: int = 0
    train_images: int = 0
    val_images: int = 0
    class_counter: Counter | None = None

    def __post_init__(self) -> None:
        if self.class_counter is None:
            self.class_counter = Counter()


def ensure_layout(out_dir: Path) -> None:
    for split in ['train', 'val']:
        (out_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)


def stable_split(key: str, val_ratio: float) -> str:
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(key)) % 10000
    random_value = (seed % 1000) / 1000.0
    return 'val' if random_value < val_ratio else 'train'


def maybe_resize(image: np.ndarray, mask: np.ndarray, max_side: int) -> tuple[np.ndarray, np.ndarray]:
    if max_side <= 0:
        return image, mask

    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image, mask

    scale = max_side / float(longest)
    new_w, new_h = max(32, int(w * scale)), max(32, int(h * scale))

    image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return image_resized, mask_resized


def mask_to_yolo_polygons(mask_binary: np.ndarray, class_id: int, min_area: float = 32.0) -> list[str]:
    contours, _ = cv2.findContours(mask_binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask_binary.shape[:2]

    lines: list[str] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape(-1, 2)
        if len(points) < 3:
            continue

        normalized = []
        for x, y in points:
            normalized.extend([float(x) / w, float(y) / h])

        if len(normalized) < 6:
            continue
        lines.append(f"{class_id} " + ' '.join(f'{v:.6f}' for v in normalized))

    return lines


def encode_label_lines(mask: np.ndarray, source_to_target: dict[int, list[int]], min_area: float = 32.0) -> list[str]:
    lines: list[str] = []
    for target_class_id, source_values in source_to_target.items():
        merged = np.isin(mask, source_values).astype(np.uint8)
        if merged.sum() == 0:
            continue

        poly_lines = mask_to_yolo_polygons(merged, class_id=target_class_id, min_area=min_area)
        lines.extend(poly_lines)
    return lines


def write_sample(
    out_dir: Path,
    split: str,
    sample_id: str,
    image_bgr: np.ndarray,
    label_lines: list[str],
) -> None:
    img_path = out_dir / 'images' / split / f'{sample_id}.jpg'
    lbl_path = out_dir / 'labels' / split / f'{sample_id}.txt'
    cv2.imwrite(str(img_path), image_bgr)
    lbl_path.write_text('\n'.join(label_lines), encoding='utf-8')


def process_chronoroot2(
    out_dir: Path,
    max_samples: int,
    val_ratio: float,
    img_max_side: int,
    seed: int,
    min_area: float,
) -> BuildStats:
    stats = BuildStats()
    api = HfApi()
    files = [
        f for f in api.list_repo_files(repo_id='ngaggion/ChronoRoot2', repo_type='dataset') if f.endswith('.nii.gz')
    ]

    rng = random.Random(seed)
    if max_samples > 0 and len(files) > max_samples:
        files = rng.sample(files, max_samples)

    for idx, nii_rel in enumerate(files):
        stats.total_images += 1
        png_rel = nii_rel.replace('.nii.gz', '.png')

        try:
            local_nii = hf_hub_download(
                repo_id='ngaggion/ChronoRoot2',
                repo_type='dataset',
                filename=nii_rel,
            )
            local_png = hf_hub_download(
                repo_id='ngaggion/ChronoRoot2',
                repo_type='dataset',
                filename=png_rel,
            )
        except Exception:
            continue

        image_gray = cv2.imread(local_png, cv2.IMREAD_GRAYSCALE)
        if image_gray is None:
            continue

        mask = nib.load(local_nii).get_fdata().astype(np.uint8)
        if mask.ndim != 2:
            mask = np.squeeze(mask)
        if mask.ndim != 2:
            continue

        image_gray, mask = maybe_resize(image_gray, mask, img_max_side)
        image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

        label_lines = encode_label_lines(mask, CHRONOROOT_TO_TARGET, min_area=min_area)
        if not label_lines:
            stats.empty_masks += 1
            continue

        split = stable_split(f'chrono_{nii_rel}', val_ratio)
        sample_id = f'chrono_{idx:06d}'
        write_sample(out_dir, split, sample_id, image_bgr, label_lines)

        stats.written_images += 1
        if split == 'train':
            stats.train_images += 1
        else:
            stats.val_images += 1

        for line in label_lines:
            cls = int(line.split()[0])
            stats.class_counter[CLASS_NAMES[cls]] += 1

    return stats


def process_plantorgans(
    out_dir: Path,
    max_samples: int,
    img_max_side: int,
    seed: int,
    min_area: float,
) -> BuildStats:
    del seed
    stats = BuildStats()

    processed = 0
    split_map = {'train': 'train', 'validation': 'val'}
    for hf_split, out_split in split_map.items():
        ds = load_dataset('farmaieu/plantorgans', split=hf_split, streaming=True)

        for item in ds:
            if max_samples > 0 and processed >= max_samples:
                break

            stats.total_images += 1
            processed += 1

            image = np.array(item['image'].convert('RGB'))
            label = np.array(item['label'])
            if label.ndim == 3:
                label = label[..., 0]

            image, label = maybe_resize(image, label, img_max_side)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            label_lines = encode_label_lines(label.astype(np.uint8), PLANTORGANS_TO_TARGET, min_area=min_area)
            if not label_lines:
                stats.empty_masks += 1
                continue

            sample_id = f'plantorg_{hf_split}_{processed:06d}'
            write_sample(out_dir, out_split, sample_id, image_bgr, label_lines)

            stats.written_images += 1
            if out_split == 'train':
                stats.train_images += 1
            else:
                stats.val_images += 1

            for line in label_lines:
                cls = int(line.split()[0])
                stats.class_counter[CLASS_NAMES[cls]] += 1

        if max_samples > 0 and processed >= max_samples:
            break

    return stats


def process_weak_100_crops(
    out_dir: Path,
    max_samples: int,
    val_ratio: float,
    seed: int,
) -> BuildStats:
    stats = BuildStats()

    zip_name = 'leaflogic object detection.v5i.yolov5pytorch.zip'
    local_zip = hf_hub_download(
        repo_id='devshaheen/100_crops_plants_object_detection_25k_image_dataset',
        repo_type='dataset',
        filename=zip_name,
    )

    with zipfile.ZipFile(local_zip, 'r') as zf:
        members = zf.namelist()
        label_members = sorted(
            m for m in members if '/labels/' in m.replace('\\', '/') and m.lower().endswith('.txt')
        )
        rng = random.Random(seed)
        if max_samples > 0 and len(label_members) > max_samples:
            label_members = rng.sample(label_members, max_samples)

        for idx, lbl_member in enumerate(label_members):
            stats.total_images += 1
            normalized = lbl_member.replace('\\', '/')
            stem = Path(normalized).stem

            image_candidates = [
                normalized.replace('/labels/', '/images/').rsplit('.', 1)[0] + '.jpg',
                normalized.replace('/labels/', '/images/').rsplit('.', 1)[0] + '.jpeg',
                normalized.replace('/labels/', '/images/').rsplit('.', 1)[0] + '.png',
                normalized.replace('/labels/', '/images/').rsplit('.', 1)[0] + '.JPG',
            ]
            img_member = next((m for m in image_candidates if m in members), None)
            if img_member is None:
                continue

            try:
                img_bytes = zf.read(img_member)
                label_text = zf.read(lbl_member).decode('utf-8', errors='ignore')
            except Exception:
                continue

            image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue

            lines = []
            for row in label_text.splitlines():
                parts = row.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    _, xc, yc, bw, bh = map(float, parts)
                except ValueError:
                    continue
                x1 = max(0.0, xc - bw / 2.0)
                y1 = max(0.0, yc - bh / 2.0)
                x2 = min(1.0, xc + bw / 2.0)
                y2 = min(1.0, yc + bh / 2.0)
                # Weak segmentation: bbox rectangle as polygon, mapped to leaves class.
                lines.append(f"2 {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}")

            if not lines:
                stats.empty_masks += 1
                continue

            split = stable_split(f'weak_{stem}_{idx}', val_ratio)
            sample_id = f'weak100_{idx:06d}'
            write_sample(out_dir, split, sample_id, image, lines)

            stats.written_images += 1
            if split == 'train':
                stats.train_images += 1
            else:
                stats.val_images += 1
            stats.class_counter['leaves'] += len(lines)
    return stats


def process_plantseg_lesions(
    out_dir: Path,
    max_samples: int,
    val_ratio: float,
) -> BuildStats:
    stats = BuildStats()
    samples_json = hf_hub_download(
        repo_id='Voxel51/PlantSeg-Test',
        repo_type='dataset',
        filename='samples.json',
    )

    seen = 0
    with Path(samples_json).open('rb') as fh:
        for sample in ijson.items(fh, 'samples.item'):
            if max_samples > 0 and seen >= max_samples:
                break

            stats.total_images += 1
            seen += 1

            filepath = sample.get('filepath')
            if not filepath:
                continue

            try:
                local_image = hf_hub_download(
                    repo_id='Voxel51/PlantSeg-Test',
                    repo_type='dataset',
                    filename=filepath,
                )
            except Exception:
                continue

            image = cv2.imread(local_image, cv2.IMREAD_COLOR)
            if image is None:
                continue

            poly_container = (sample.get('segmentations') or {}).get('polylines', [])
            lines = []
            for poly in poly_container:
                points_nested = poly.get('points') or []
                if not points_nested:
                    continue
                pts = points_nested[0]
                if not isinstance(pts, list) or len(pts) < 3:
                    continue

                coords = []
                for point in pts:
                    if not isinstance(point, list) or len(point) != 2:
                        continue
                    coords.extend([float(point[0]), float(point[1])])
                if len(coords) < 6:
                    continue

                # Lesion polygons are mapped as weak leaf regions.
                lines.append('2 ' + ' '.join(f'{v:.6f}' for v in coords))

            if not lines:
                stats.empty_masks += 1
                continue

            split = stable_split(f'plantseg_{filepath}', val_ratio)
            sample_id = f'plantseg_{seen:06d}'
            write_sample(out_dir, split, sample_id, image, lines)

            stats.written_images += 1
            if split == 'train':
                stats.train_images += 1
            else:
                stats.val_images += 1
            stats.class_counter['leaves'] += len(lines)

    return stats


def write_dataset_yaml(out_dir: Path) -> Path:
    dataset_yaml = {
        'path': str(out_dir.resolve()).replace('\\', '/'),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    target = out_dir / 'dataset.yaml'
    with target.open('w', encoding='utf-8') as fh:
        yaml.safe_dump(dataset_yaml, fh, sort_keys=False, allow_unicode=False)
    return target


def merge_stats(stats_by_source: dict[str, BuildStats]) -> dict:
    merged = {
        'sources': {},
        'total_written': 0,
        'total_train': 0,
        'total_val': 0,
        'class_instances': Counter(),
    }

    for source, st in stats_by_source.items():
        merged['sources'][source] = {
            'total_seen': st.total_images,
            'written': st.written_images,
            'empty_masks': st.empty_masks,
            'train': st.train_images,
            'val': st.val_images,
            'class_instances': dict(st.class_counter),
        }
        merged['total_written'] += st.written_images
        merged['total_train'] += st.train_images
        merged['total_val'] += st.val_images
        merged['class_instances'].update(st.class_counter)

    merged['class_instances'] = dict(merged['class_instances'])
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare multi-source plant dataset in YOLO-seg format.')
    parser.add_argument('--out', default='data/hf_multisource_yoloseg', type=str)
    parser.add_argument('--chronoroot-max', default=1200, type=int)
    parser.add_argument('--plantorgans-max', default=4000, type=int)
    parser.add_argument('--include-weak-100crops', action='store_true')
    parser.add_argument('--weak-100crops-max', default=2500, type=int)
    parser.add_argument('--include-plantseg-lesions', action='store_true')
    parser.add_argument('--plantseg-max', default=1000, type=int)
    parser.add_argument('--val-ratio', default=0.15, type=float)
    parser.add_argument('--img-max-side', default=1280, type=int)
    parser.add_argument('--min-area', default=64.0, type=float)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_layout(out_dir)

    stats_by_source: dict[str, BuildStats] = {}

    print('Processing ChronoRoot2...')
    stats_by_source['chronoroot2'] = process_chronoroot2(
        out_dir=out_dir,
        max_samples=args.chronoroot_max,
        val_ratio=args.val_ratio,
        img_max_side=args.img_max_side,
        seed=args.seed,
        min_area=args.min_area,
    )

    print('Processing PlantOrgans...')
    stats_by_source['plantorgans'] = process_plantorgans(
        out_dir=out_dir,
        max_samples=args.plantorgans_max,
        img_max_side=args.img_max_side,
        seed=args.seed,
        min_area=args.min_area,
    )

    if args.include_weak_100crops:
        print('Processing weak 100 Crops bbox -> polygon conversion...')
        stats_by_source['weak_100crops'] = process_weak_100_crops(
            out_dir=out_dir,
            max_samples=args.weak_100crops_max,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

    if args.include_plantseg_lesions:
        print('Processing PlantSeg-Test lesion polygons (weak leaf proxy)...')
        stats_by_source['plantseg_lesions'] = process_plantseg_lesions(
            out_dir=out_dir,
            max_samples=args.plantseg_max,
            val_ratio=args.val_ratio,
        )

    dataset_yaml_path = write_dataset_yaml(out_dir)
    merged = merge_stats(stats_by_source)
    merged['dataset_yaml'] = str(dataset_yaml_path)

    (out_dir / 'build_stats.json').write_text(
        json.dumps(merged, indent=2, ensure_ascii=True),
        encoding='utf-8',
    )

    print('Dataset ready:', dataset_yaml_path)
    print(json.dumps(merged, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
