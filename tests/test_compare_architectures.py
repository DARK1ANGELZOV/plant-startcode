import json
from pathlib import Path

from training.compare_architectures import compare_architectures


def test_compare_architectures_selects_higher_score(tmp_path: Path) -> None:
    yolo_report = {
        'clean': {'miou': 0.62, 'precision': 0.7, 'recall': 0.68, 'boundary_iou': 0.41},
        'robustness': {'robustness_score': 0.55, 'mean_drop': 0.08},
        'best_checkpoint': 'models/robust/yolo_robust_best.pt',
        'onnx_path': 'models/robust/yolo_robust_best.onnx',
    }
    deeplab_report = {
        'clean': {'miou': 0.58, 'precision': 0.66, 'recall': 0.64, 'boundary_iou': 0.49},
        'robustness': {'robustness_score': 0.57, 'mean_drop': 0.05},
        'checkpoint': 'models/robust/deeplab_robust_final.pt',
    }

    yolo_path = tmp_path / 'yolo.json'
    deeplab_path = tmp_path / 'deeplab.json'
    output_path = tmp_path / 'best.json'

    yolo_path.write_text(json.dumps(yolo_report), encoding='utf-8')
    deeplab_path.write_text(json.dumps(deeplab_report), encoding='utf-8')

    result = compare_architectures(
        yolo_report_path=str(yolo_path),
        deeplab_report_path=str(deeplab_path),
        output_path=str(output_path),
    )

    assert result['best_architecture'] in {'yolo_seg', 'deeplabv3'}
    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding='utf-8'))
    assert 'candidates' in loaded
    assert len(loaded['candidates']) == 2
