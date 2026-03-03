import pytest

torch = pytest.importorskip('torch')

from training.robust_corruptions import (
    additive_noise,
    fog,
    gaussian_blur,
    get_corruption_fn,
    jpeg_artifacts,
    motion_blur,
    rain,
)


@pytest.mark.parametrize(
    'fn',
    [
        gaussian_blur,
        motion_blur,
        additive_noise,
        jpeg_artifacts,
        fog,
        rain,
    ],
)
def test_corruptions_preserve_shape_and_range(fn) -> None:
    images = torch.rand(2, 3, 64, 64)
    out = fn(images)
    assert out.shape == images.shape
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_get_corruption_fn_known_names() -> None:
    images = torch.rand(1, 3, 32, 32)
    for name in ['gaussian_blur', 'motion_blur', 'noise', 'jpeg', 'fog', 'rain']:
        fn = get_corruption_fn(name)
        out = fn(images)
        assert out.shape == images.shape


def test_get_corruption_fn_unknown_name() -> None:
    with pytest.raises(ValueError):
        get_corruption_fn('unknown_corruption')
