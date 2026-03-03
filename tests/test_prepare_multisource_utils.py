import numpy as np

from training.prepare_multisource_dataset import (
    encode_label_lines,
    stable_split,
)


def test_stable_split_deterministic() -> None:
    key = 'chrono_example'
    first = stable_split(key, 0.2)
    second = stable_split(key, 0.2)
    assert first == second
    assert first in {'train', 'val'}


def test_encode_label_lines_generates_polygons() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:30, 10:30] = 1  # root source class
    mask[35:55, 8:26] = 4   # stem source class
    mask[20:40, 36:58] = 5  # leaf source class

    source_to_target = {
        0: [1],
        1: [4],
        2: [5],
    }
    lines = encode_label_lines(mask, source_to_target, min_area=20)
    assert len(lines) >= 3
    assert any(line.startswith('0 ') for line in lines)
    assert any(line.startswith('1 ') for line in lines)
    assert any(line.startswith('2 ') for line in lines)
