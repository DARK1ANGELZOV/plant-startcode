import numpy as np

from morphometry.analysis import analyze_mask, area_pixels, longest_path_length


def test_area_pixels_square() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    assert area_pixels(mask) == 100


def test_longest_path_line() -> None:
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:40, 25] = 1
    length = longest_path_length(mask)
    assert length >= 28


def test_analyze_mask_values() -> None:
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[5:25, 10:20] = 1
    result = analyze_mask(mask, mm_per_px=0.1)
    assert result.area_px == 200
    assert result.area_mm2 > 1.9
    assert result.length_mm > 1.0
