import numpy as np
import pytest

torch = pytest.importorskip('torch')

from training.robust_metrics import boundary_iou, compute_seg_metrics, robustness_score


def test_compute_seg_metrics_shapes_and_keys() -> None:
    confmat = torch.tensor(
        [
            [120, 0, 0, 0],
            [0, 40, 10, 0],
            [0, 5, 30, 5],
            [0, 0, 10, 20],
        ],
        dtype=torch.float64,
    )
    class_names = ['background', 'root', 'stem', 'leaves']
    metrics = compute_seg_metrics(confmat, class_names=class_names, boundary_scores=[0.8, 0.9])

    assert 0.0 <= metrics.miou <= 1.0
    assert 0.0 <= metrics.mdice <= 1.0
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.boundary_iou <= 1.0
    assert set(metrics.per_class_iou.keys()) == {'root', 'stem', 'leaves'}


def test_boundary_iou_perfect_prediction() -> None:
    target = np.zeros((32, 32), dtype=np.uint8)
    target[5:25, 5:25] = 1
    pred = target.copy()
    score = boundary_iou(pred, target, num_classes=2, dilation=2)
    assert score > 0.99


def test_robustness_score_drop_is_non_negative() -> None:
    clean = type('Metrics', (), {'miou': 0.8})()
    corrupted = {
        'fog': type('Metrics', (), {'miou': 0.65})(),
        'rain': type('Metrics', (), {'miou': 0.7})(),
    }
    report = robustness_score(clean, corrupted)

    assert report['clean_miou'] == 0.8
    assert report['drop_by_corruption']['fog'] >= 0.0
    assert report['drop_by_corruption']['rain'] >= 0.0
    assert report['mean_drop'] >= 0.0
    assert report['robustness_score'] >= 0.0
