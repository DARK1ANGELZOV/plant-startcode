from __future__ import annotations

from collections import Counter
from pathlib import Path

from training.filter_balance_yoloseg import SampleMeta, _aggregate_instances, _balance_by_pruning, _max_ratio


def _sample(tmp_path: Path, split: str, name: str, class_instances: dict[int, int]) -> SampleMeta:
    p = tmp_path / f'{name}.jpg'
    l = tmp_path / f'{name}.txt'
    p.write_bytes(b'img')
    l.write_text('', encoding='utf-8')
    return SampleMeta(
        split=split,
        image_path=p,
        label_path=l,
        blur_score=100.0,
        brightness=120.0,
        total_area_ratio=0.1,
        class_instances=Counter(class_instances),
        class_area_px=Counter({k: float(v * 100) for k, v in class_instances.items()}),
    )


def test_balance_by_pruning_reduces_max_ratio(tmp_path: Path) -> None:
    samples = []
    for i in range(18):
        samples.append(_sample(tmp_path, 'train', f'leaf_{i}', {2: 6}))
    for i in range(6):
        samples.append(_sample(tmp_path, 'train', f'root_{i}', {0: 3}))
    for i in range(5):
        samples.append(_sample(tmp_path, 'train', f'stem_{i}', {1: 3}))

    before = _aggregate_instances(samples)
    before_ratio = _max_ratio(before)
    kept = _balance_by_pruning(samples, class_ratio_max=2.2, min_kept_images=12)
    after = _aggregate_instances(kept)
    after_ratio = _max_ratio(after)

    assert before_ratio > after_ratio
    assert len(kept) >= 12
    assert after_ratio <= 2.2 + 0.35


def test_max_ratio_empty_counter() -> None:
    assert _max_ratio(Counter()) == 0.0

