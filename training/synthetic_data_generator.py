from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def _random_shadow(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    x1 = random.randint(0, max(1, w - 1))
    x2 = random.randint(0, max(1, w - 1))
    poly = np.array([[x1, 0], [x2, h], [w, h], [w, 0]], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    alpha = random.uniform(0.15, 0.45)
    out[mask > 0] = np.clip(out[mask > 0].astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)
    return out


def _random_rain(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    layer = np.zeros_like(out)
    drops = random.randint(400, 1200)
    for _ in range(drops):
        x = random.randint(0, max(1, w - 1))
        y = random.randint(0, max(1, h - 1))
        ln = random.randint(6, 16)
        cv2.line(layer, (x, y), (min(w - 1, x + 2), min(h - 1, y + ln)), (190, 190, 190), 1)
    return cv2.addWeighted(out, 1.0, layer, 0.25, 0)


def _random_fog(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    y = np.linspace(-1, 1, h).reshape(h, 1)
    x = np.linspace(-1, 1, w).reshape(1, w)
    dist = np.sqrt(x * x + y * y)
    mask = np.clip(1.0 - dist, 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=15)
    fog = np.full_like(img, 230)
    strength = random.uniform(0.2, 0.45)
    out = img.astype(np.float32) * (1 - strength * mask[..., None]) + fog.astype(np.float32) * (strength * mask[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def _low_light(img: np.ndarray) -> np.ndarray:
    gamma = random.uniform(1.4, 2.3)
    inv_gamma = 1.0 / gamma
    lut = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype=np.uint8)
    dark = cv2.LUT(img, lut)
    return np.clip(dark.astype(np.float32) * random.uniform(0.55, 0.85), 0, 255).astype(np.uint8)


def _jpeg_artifacts(img: np.ndarray) -> np.ndarray:
    quality = random.randint(18, 45)
    ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return dec if dec is not None else img


def _apply_synthetic_transform(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    # Illumination and camera effects.
    alpha = random.uniform(0.7, 1.3)
    beta = random.uniform(-20, 20)
    out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    if random.random() < 0.6:
        out = cv2.GaussianBlur(out, (random.choice([3, 5, 7]), random.choice([3, 5, 7])), 0)
    if random.random() < 0.4:
        noise = np.random.normal(0, random.uniform(4, 22), size=out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.35:
        out = _random_shadow(out)
    if random.random() < 0.25:
        out = _random_rain(out)
    if random.random() < 0.25:
        out = _random_fog(out)
    if random.random() < 0.25:
        out = _low_light(out)
    if random.random() < 0.35:
        out = _jpeg_artifacts(out)
    return out


def _load_dataset_yaml(path: str) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
    if not isinstance(payload, dict):
        raise ValueError('Invalid dataset yaml.')
    return payload


def generate_synthetic_dataset(
    data_yaml: str,
    out_dir: str,
    copies_per_image: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)

    cfg = _load_dataset_yaml(data_yaml)
    root = Path(cfg.get('path', Path(data_yaml).parent))
    if not root.is_absolute():
        root = (Path(data_yaml).parent / root).resolve()

    out = Path(out_dir)
    out_images_train = out / 'images' / 'train'
    out_images_val = out / 'images' / 'val'
    out_labels_train = out / 'labels' / 'train'
    out_labels_val = out / 'labels' / 'val'
    for p in (out_images_train, out_images_val, out_labels_train, out_labels_val):
        p.mkdir(parents=True, exist_ok=True)

    def _copy_split(split_name: str, out_img_dir: Path, out_lbl_dir: Path) -> int:
        src_img_dir = root / str(cfg.get(split_name, f'images/{split_name}'))
        src_lbl_dir = root / 'labels' / split_name
        count = 0
        for img_path in sorted(src_img_dir.rglob('*')):
            if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
                continue
            lbl_path = src_lbl_dir / f'{img_path.stem}.txt'
            if not lbl_path.exists():
                continue
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image is None:
                continue

            # Keep original sample too.
            dst_img = out_img_dir / f'{img_path.stem}_orig{img_path.suffix.lower()}'
            dst_lbl = out_lbl_dir / f'{img_path.stem}_orig.txt'
            cv2.imwrite(str(dst_img), image)
            shutil.copy2(lbl_path, dst_lbl)
            count += 1

            for i in range(copies_per_image):
                syn = _apply_synthetic_transform(image)
                syn_img = out_img_dir / f'{img_path.stem}_syn{i:02d}{img_path.suffix.lower()}'
                syn_lbl = out_lbl_dir / f'{img_path.stem}_syn{i:02d}.txt'
                cv2.imwrite(str(syn_img), syn)
                shutil.copy2(lbl_path, syn_lbl)
                count += 1
        return count

    train_count = _copy_split('train', out_images_train, out_labels_train)
    val_count = _copy_split('val', out_images_val, out_labels_val)

    out_yaml = {
        'path': str(out.resolve()).replace('\\', '/'),
        'train': 'images/train',
        'val': 'images/val',
        'names': cfg.get('names', {0: 'root', 1: 'stem', 2: 'leaves'}),
    }
    out_yaml_path = out / 'dataset.yaml'
    out_yaml_path.write_text(yaml.safe_dump(out_yaml, sort_keys=False), encoding='utf-8')

    stats = {
        'output_dataset_yaml': str(out_yaml_path.resolve()),
        'train_images': train_count,
        'val_images': val_count,
        'copies_per_image': copies_per_image,
        'seed': seed,
    }
    (out / 'synthetic_stats.json').write_text(json.dumps(stats, indent=2, ensure_ascii=True), encoding='utf-8')
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate synthetic plant segmentation data from YOLO-seg dataset.')
    parser.add_argument('--data', default='data/hf_multisource_medium/dataset.yaml')
    parser.add_argument('--out', default='data/synthetic/generated')
    parser.add_argument('--copies-per-image', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    stats = generate_synthetic_dataset(
        data_yaml=args.data,
        out_dir=args.out,
        copies_per_image=max(0, args.copies_per_image),
        seed=args.seed,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
