from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from calibration.chessboard import ScaleCalibrator
from services.inference_service import InferenceService
from services.model_service import ModelService
from services.recommendation_service import RecommendationService
from services.report_service import ReportService
from services.storage_service import StorageService
from utils.config import load_app_config
from utils.logging import setup_logging
from utils.seed import set_global_seed


async def run_demo(
    image_path: str,
    crop: str,
    calibration_path: str | None,
    conf: float | None,
    iou: float | None,
    max_det: int | None,
) -> None:
    config = load_app_config('configs/app.yaml')
    setup_logging(config['app'].get('log_level', 'INFO'))
    set_global_seed(int(config.get('seed', {}).get('value', 42)))

    model_service = ModelService(config)
    await model_service.load()

    calibrator = ScaleCalibrator(
        cache_path=config['calibration']['cache_path'],
        default_mm_per_px=float(config['morphometry']['default_mm_per_px']),
        board_size=tuple(config['calibration']['board_size']),
        square_size_mm=float(config['calibration']['square_size_mm']),
    )

    inference_service = InferenceService(
        model_service=model_service,
        calibrator=calibrator,
        storage=StorageService(config['inference']['output_root']),
        reporter=ReportService(),
        recommender=RecommendationService(config['morphometry'].get('recommendation_thresholds', {})),
        config=config,
    )

    image_bytes = Path(image_path).read_bytes()
    calibration_bytes = Path(calibration_path).read_bytes() if calibration_path else None

    result = await inference_service.run_single(
        image_bytes=image_bytes,
        image_name=Path(image_path).name,
        crop=crop,
        calibration_bytes=calibration_bytes,
        camera_id='demo_cam',
        conf=conf,
        iou=iou,
        max_det=max_det,
    )

    print('Run ID:', result.run_id)
    print('Scale:', result.scale_mm_per_px, '| source:', result.scale_source)
    print('Measurements:', len(result.measurements))
    print('Files:', result.files)
    for rec in result.recommendations:
        print(f"- [{rec.severity}] {rec.message} | {rec.action}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Run demo inference for Agro AI System.')
    parser.add_argument('--image', required=True, type=str)
    parser.add_argument('--crop', default='Wheat', type=str)
    parser.add_argument('--calibration', default=None, type=str)
    parser.add_argument('--conf', default=0.05, type=float)
    parser.add_argument('--iou', default=0.5, type=float)
    parser.add_argument('--max-det', default=40, type=int)
    args = parser.parse_args()

    asyncio.run(
        run_demo(
            args.image,
            args.crop,
            args.calibration,
            args.conf,
            args.iou,
            args.max_det,
        )
    )


if __name__ == '__main__':
    main()
