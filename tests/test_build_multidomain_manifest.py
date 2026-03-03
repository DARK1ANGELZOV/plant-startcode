import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from training.build_multidomain_manifest import build_manifest


def test_build_manifest_from_minimal_yolo_seg(tmp_path: Path) -> None:
    ds_root = tmp_path / 'plants_local'
    img_dir = ds_root / 'images' / 'train'
    lbl_dir = ds_root / 'labels' / 'train'
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[10:40, 10:40] = (0, 255, 0)
    image_path = img_dir / 'sample_001.png'
    cv2.imwrite(str(image_path), image)

    # YOLO-seg polygon for class 0 (root), square contour.
    label_path = lbl_dir / 'sample_001.txt'
    label_path.write_text(
        '0 0.15625 0.15625 0.625 0.15625 0.625 0.625 0.15625 0.625\n',
        encoding='utf-8',
    )

    dataset_yaml = {
        'path': str(ds_root).replace('\\', '/'),
        'train': 'images/train',
        'val': 'images/train',
        'names': {0: 'root', 1: 'stem', 2: 'leaves'},
    }
    (ds_root / 'dataset.yaml').write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding='utf-8')

    config = {
        'datasets': {
            'plants_local': {
                'root': str(ds_root),
                'split': 'train',
                'type': 'yolo_seg',
                'usage': 'plant',
                'has_labels': True,
            }
        },
        'mapping': {
            'default_unknown': -1,
            'by_dataset': {
                'plants_local': {
                    'root': 1,
                    'stem': 2,
                    'leaves': 3,
                }
            },
        },
    }
    cfg_path = tmp_path / 'robust_test.yaml'
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')

    out_manifest = tmp_path / 'robust' / 'train_manifest.jsonl'
    info = build_manifest(str(cfg_path), str(out_manifest), split='train', max_images_per_domain=10)

    assert info['total_rows'] == 1
    assert out_manifest.exists()

    line = out_manifest.read_text(encoding='utf-8').strip()
    row = json.loads(line)
    assert row['domain'] == 'plants_local'
    assert Path(row['mask_path']).exists()

    mask = cv2.imread(row['mask_path'], cv2.IMREAD_UNCHANGED)
    assert mask is not None
    assert int(mask.max()) == 1  # root is mapped to class index 1 (background is 0)
