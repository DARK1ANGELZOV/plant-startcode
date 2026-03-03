import numpy as np

from services.disease_service import DiseaseService


class _Det:
    def __init__(self, class_name: str, mask: np.ndarray):
        self.class_name = class_name
        self.mask = mask


def test_disease_service_detects_chlorosis_and_returns_actions() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Yellow-ish patch in BGR
    img[:, :] = (0, 255, 255)

    mask = np.ones((100, 100), dtype=np.uint8)
    dets = [_Det('leaves', mask)]

    out = DiseaseService().analyze(img, dets, measurements=[])
    assert out['risk_level'] in {'medium', 'high', 'low'}
    assert out['confidence'] >= 0.0
    assert out['actions']


def test_disease_service_without_leaf_masks_returns_unknown() -> None:
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    out = DiseaseService().analyze(img, detections=[], measurements=[])
    assert out['risk_level'] == 'unknown'
