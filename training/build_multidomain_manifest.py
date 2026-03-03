from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml

from utils.config import load_yaml


TARGET_CLASS_NAMES = ['background', 'root', 'stem', 'leaves']
IGNORE_INDEX = 255


@dataclass
class ManifestRow:
    image_path: str
    mask_path: str
    domain: str
    usage: str
    split: str
    is_adverse: bool
    has_labels: bool


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _normalize_class_mapping(raw: dict[str, Any]) -> tuple[dict[int, int], dict[str, int]]:
    by_id: dict[int, int] = {}
    by_name: dict[str, int] = {}
    for key, value in (raw or {}).items():
        if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
            by_id[int(key)] = int(value)
        else:
            by_name[str(key).lower()] = int(value)
    return by_id, by_name


def _write_mask(mask: np.ndarray, out_path: Path) -> None:
    _ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), mask.astype(np.uint8))


def _polygon_to_mask(h: int, w: int, points: list[float]) -> np.ndarray:
    pts = np.array(points, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    pts_i = pts.astype(np.int32)
    out = np.zeros((h, w), dtype=np.uint8)
    if len(pts_i) >= 3:
        cv2.fillPoly(out, [pts_i], 1)
    return out


def _load_dataset_names(root: Path) -> dict[int, str]:
    ds_yaml = root / 'dataset.yaml'
    if not ds_yaml.exists():
        return {}
    payload = yaml.safe_load(ds_yaml.read_text(encoding='utf-8')) or {}
    names = payload.get('names', {})
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {idx: str(v) for idx, v in enumerate(names)}
    return {}


def _resolve_target_class(
    source_id: int,
    source_name: str | None,
    by_id: dict[int, int],
    by_name: dict[str, int],
    default_unknown: int,
    yolo_mode: bool = False,
) -> int:
    if source_id in by_id:
        return by_id[source_id]
    if source_name and source_name.lower() in by_name:
        return by_name[source_name.lower()]

    if yolo_mode:
        # YOLO root/stem/leaves labels are often 0/1/2 -> shift by +1 for background at 0.
        shifted = source_id + 1
        if 0 <= shifted <= 3:
            return shifted

    return default_unknown


def _collect_yolo_seg(
    domain: str,
    root: Path,
    split: str,
    usage: str,
    is_adverse: bool,
    class_map_raw: dict[str, Any],
    default_unknown: int,
    cache_dir: Path,
    max_images: int,
) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    image_dir = root / 'images' / split
    label_dir = root / 'labels' / split
    if not image_dir.exists() or not label_dir.exists():
        return rows

    names_map = _load_dataset_names(root)
    by_id, by_name = _normalize_class_mapping(class_map_raw)

    for img_path in sorted(image_dir.rglob('*')):
        if max_images > 0 and len(rows) >= max_images:
            break
        if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
            continue
        lbl_path = label_dir / f'{img_path.stem}.txt'
        if not lbl_path.exists():
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        lines = lbl_path.read_text(encoding='utf-8', errors='ignore').splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:
                continue

            src_id = int(float(parts[0]))
            src_name = names_map.get(src_id)
            target_class = _resolve_target_class(
                src_id,
                src_name,
                by_id,
                by_name,
                default_unknown,
                yolo_mode=True,
            )
            if target_class < 0:
                continue

            coords = [float(x) for x in parts[1:]]
            if len(coords) % 2 != 0:
                continue
            pts = []
            for i in range(0, len(coords), 2):
                pts.extend([coords[i] * w, coords[i + 1] * h])
            poly_mask = _polygon_to_mask(h, w, pts)
            mask[poly_mask > 0] = np.uint8(target_class)

        if (mask > 0).sum() == 0:
            continue

        out_mask = cache_dir / domain / split / f'{img_path.stem}.png'
        _write_mask(mask, out_mask)
        rows.append(
            ManifestRow(
                image_path=str(img_path.resolve()),
                mask_path=str(out_mask.resolve()),
                domain=domain,
                usage=usage,
                split=split,
                is_adverse=is_adverse,
                has_labels=True,
            )
        )

    return rows


def _collect_generic_semseg(
    domain: str,
    root: Path,
    split: str,
    usage: str,
    is_adverse: bool,
    class_map_raw: dict[str, Any],
    default_unknown: int,
    cache_dir: Path,
    max_images: int,
) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    img_candidates = [
        root / 'images' / split,
        root / split / 'images',
        root / 'rgb' / split,
        root / split,
    ]
    mask_candidates = [
        root / 'masks' / split,
        root / split / 'masks',
        root / 'labels' / split,
        root / 'gt' / split,
    ]

    image_dir = next((p for p in img_candidates if p.exists()), None)
    mask_dir = next((p for p in mask_candidates if p.exists()), None)
    if image_dir is None or mask_dir is None:
        return rows

    by_id, by_name = _normalize_class_mapping(class_map_raw)

    mask_by_stem: dict[str, Path] = {}
    for mp in mask_dir.rglob('*'):
        if mp.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}:
            mask_by_stem[mp.stem] = mp

    for img_path in sorted(image_dir.rglob('*')):
        if max_images > 0 and len(rows) >= max_images:
            break
        if img_path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.bmp'}:
            continue
        mask_path = mask_by_stem.get(img_path.stem)
        if mask_path is None:
            continue

        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask_raw is None:
            continue
        if mask_raw.ndim == 3:
            mask_raw = mask_raw[..., 0]

        mapped = np.full(mask_raw.shape, IGNORE_INDEX, dtype=np.uint8)
        uniq = np.unique(mask_raw)

        any_mapped = False
        for uid in uniq:
            src_id = int(uid)
            target = by_id.get(src_id)
            if target is None:
                # fallback: any non-zero class is treated as leaves
                if src_id > 0:
                    target = 3 if default_unknown < 0 else default_unknown
                else:
                    target = 0
            if target < 0:
                continue
            mapped[mask_raw == src_id] = np.uint8(target)
            any_mapped = any_mapped or (target > 0)

        if not any_mapped:
            continue

        out_mask = cache_dir / domain / split / f'{img_path.stem}.png'
        _write_mask(mapped, out_mask)
        rows.append(
            ManifestRow(
                image_path=str(img_path.resolve()),
                mask_path=str(out_mask.resolve()),
                domain=domain,
                usage=usage,
                split=split,
                is_adverse=is_adverse,
                has_labels=True,
            )
        )

    return rows


def _collect_acdc(
    domain: str,
    root: Path,
    split: str,
    usage: str,
    class_map_raw: dict[str, Any],
    default_unknown: int,
    cache_dir: Path,
    max_images: int,
) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    gt_root = root / 'gt' / split
    rgb_root = root / 'rgb_anon' / split
    if not gt_root.exists() or not rgb_root.exists():
        return rows

    by_id, _ = _normalize_class_mapping(class_map_raw)

    for gt_path in sorted(gt_root.rglob('*_gt_labelIds.png')):
        if max_images > 0 and len(rows) >= max_images:
            break
        rel = gt_path.relative_to(gt_root)
        stem = gt_path.name.replace('_gt_labelIds.png', '_rgb_anon.png')
        img_path = (rgb_root / rel.parent / stem)
        if not img_path.exists():
            continue

        raw = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue
        mapped = np.full(raw.shape, IGNORE_INDEX, dtype=np.uint8)

        any_mapped = False
        for uid in np.unique(raw):
            src = int(uid)
            target = by_id.get(src)
            if target is None:
                # fallback common cityscapes ids: vegetation=21 -> leaves
                if src == 21:
                    target = 3
                elif src == 0:
                    target = 0
                else:
                    target = default_unknown
            if target < 0:
                continue
            mapped[raw == src] = np.uint8(target)
            any_mapped = any_mapped or (target > 0)

        if not any_mapped:
            continue

        out_mask = cache_dir / domain / split / rel.with_suffix('').name
        out_mask = out_mask.with_suffix('.png')
        _write_mask(mapped, out_mask)

        cond = rel.parts[0].lower() if rel.parts else 'adverse'
        is_adverse = cond in {'fog', 'rain', 'snow', 'night'}

        rows.append(
            ManifestRow(
                image_path=str(img_path.resolve()),
                mask_path=str(out_mask.resolve()),
                domain=domain,
                usage=usage,
                split=split,
                is_adverse=is_adverse,
                has_labels=True,
            )
        )

    return rows


def _collect_coco(
    domain: str,
    root: Path,
    split: str,
    usage: str,
    class_map_raw: dict[str, Any],
    default_unknown: int,
    cache_dir: Path,
    max_images: int,
) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    try:
        from pycocotools.coco import COCO
    except Exception:
        return rows

    ann_file = root / 'annotations' / f'instances_{split}2017.json'
    img_dir = root / f'{split}2017'
    if not ann_file.exists() or not img_dir.exists():
        return rows

    by_id, by_name = _normalize_class_mapping(class_map_raw)
    coco = COCO(str(ann_file))

    cats = coco.loadCats(coco.getCatIds())
    cat_to_target: dict[int, int] = {}
    for c in cats:
        cid = int(c['id'])
        name = str(c['name']).lower()
        target = by_id.get(cid)
        if target is None:
            target = by_name.get(name)
        if target is None:
            if 'plant' in name or 'tree' in name or 'grass' in name or 'flower' in name:
                target = 3
            else:
                target = default_unknown
        cat_to_target[cid] = int(target)

    image_ids = coco.getImgIds()
    if max_images > 0:
        image_ids = image_ids[:max_images]

    for img_id in image_ids:
        img_info = coco.loadImgs([img_id])[0]
        img_path = img_dir / img_info['file_name']
        if not img_path.exists():
            continue

        h = int(img_info['height'])
        w = int(img_info['width'])
        mask = np.zeros((h, w), dtype=np.uint8)

        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            target = cat_to_target.get(int(ann['category_id']), default_unknown)
            if target < 0:
                continue
            ann_mask = coco.annToMask(ann).astype(np.uint8)
            if ann_mask.sum() == 0:
                continue
            mask[ann_mask > 0] = np.uint8(target)

        if (mask > 0).sum() == 0:
            continue

        out_mask = cache_dir / domain / split / f"{Path(img_info['file_name']).stem}.png"
        _write_mask(mask, out_mask)
        rows.append(
            ManifestRow(
                image_path=str(img_path.resolve()),
                mask_path=str(out_mask.resolve()),
                domain=domain,
                usage=usage,
                split=split,
                is_adverse=False,
                has_labels=True,
            )
        )

    return rows


def _collect_from_dataset_cfg(
    name: str,
    ds_cfg: dict[str, Any],
    mapping_cfg: dict[str, Any],
    default_unknown: int,
    cache_dir: Path,
    max_images_per_domain: int,
) -> list[ManifestRow]:
    root = Path(str(ds_cfg.get('root', '')))
    if not root.exists():
        return []

    split = str(ds_cfg.get('split', 'train'))
    usage = str(ds_cfg.get('usage', 'plant'))
    ds_type = str(ds_cfg.get('type', 'generic'))
    is_adverse = bool(ds_cfg.get('usage') == 'adverse')
    class_map_raw = mapping_cfg.get(name, {})

    if ds_type == 'yolo_seg':
        return _collect_yolo_seg(
            domain=name,
            root=root,
            split=split,
            usage=usage,
            is_adverse=is_adverse,
            class_map_raw=class_map_raw,
            default_unknown=default_unknown,
            cache_dir=cache_dir,
            max_images=max_images_per_domain,
        )

    if ds_type == 'acdc_semseg':
        return _collect_acdc(
            domain=name,
            root=root,
            split=split,
            usage=usage,
            class_map_raw=class_map_raw,
            default_unknown=default_unknown,
            cache_dir=cache_dir,
            max_images=max_images_per_domain,
        )

    if ds_type == 'coco_panoptic':
        return _collect_coco(
            domain=name,
            root=root,
            split=split,
            usage=usage,
            class_map_raw=class_map_raw,
            default_unknown=default_unknown,
            cache_dir=cache_dir,
            max_images=max_images_per_domain,
        )

    # fallback for SA-1B/OpenImages/WeatherProof/RaidaR/MSeg local exports
    return _collect_generic_semseg(
        domain=name,
        root=root,
        split=split,
        usage=usage,
        is_adverse=is_adverse,
        class_map_raw=class_map_raw,
        default_unknown=default_unknown,
        cache_dir=cache_dir,
        max_images=max_images_per_domain,
    )


def build_manifest(
    config_path: str,
    output_jsonl: str,
    split: str = 'train',
    max_images_per_domain: int = 50000,
) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    output_path = Path(output_jsonl)
    _ensure_dir(output_path.parent)

    datasets_cfg = cfg.get('datasets', {})
    mapping = cfg.get('mapping', {})
    default_unknown = int(mapping.get('default_unknown', -1))
    by_dataset = mapping.get('by_dataset', {})

    cache_dir = output_path.parent / 'cache_masks'
    _ensure_dir(cache_dir)

    rows: list[ManifestRow] = []
    summary: dict[str, Any] = {'domains': {}, 'total': 0}

    for name, ds_cfg in datasets_cfg.items():
        ds_cfg = dict(ds_cfg or {})
        ds_cfg['split'] = split if split in {'train', 'val', 'test'} else ds_cfg.get('split', 'train')

        domain_rows = _collect_from_dataset_cfg(
            name=name,
            ds_cfg=ds_cfg,
            mapping_cfg=by_dataset,
            default_unknown=default_unknown,
            cache_dir=cache_dir,
            max_images_per_domain=max_images_per_domain,
        )

        rows.extend(domain_rows)
        summary['domains'][name] = {
            'usage': ds_cfg.get('usage', 'unknown'),
            'type': ds_cfg.get('type', 'unknown'),
            'count': len(domain_rows),
            'root': ds_cfg.get('root', ''),
        }

    with output_path.open('w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(asdict(row), ensure_ascii=True) + '\n')

    summary['total'] = len(rows)
    summary_path = output_path.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding='utf-8')

    return {
        'manifest_path': str(output_path.resolve()),
        'summary_path': str(summary_path.resolve()),
        'total_rows': len(rows),
        'summary': summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Build multi-domain manifest for robust plant segmentation.')
    parser.add_argument('--config', default='configs/robust_train.yaml', type=str)
    parser.add_argument('--output', default='data/robust/train_manifest.jsonl', type=str)
    parser.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--max-images-per-domain', default=50000, type=int)
    args = parser.parse_args()

    info = build_manifest(
        config_path=args.config,
        output_jsonl=args.output,
        split=args.split,
        max_images_per_domain=args.max_images_per_domain,
    )
    print(json.dumps(info, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
