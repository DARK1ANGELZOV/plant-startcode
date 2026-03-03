from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

from inference.ensemble_predictor import run_ensemble_inference
from inference.predictor import run_yolo_inference
from utils.errors import ModelNotLoadedError

# Workaround for Windows long-path issues: if heavy deps are installed into a short path
# (e.g. C:\ptpkgs), prepend it before importing torch/ultralytics.
_extra_site = os.getenv('AGRO_EXTRA_SITE_PACKAGES', r'C:\ptpkgs').strip()
if _extra_site and os.path.isdir(_extra_site) and _extra_site not in sys.path:
    sys.path.insert(0, _extra_site)

try:
    import torch
except Exception:
    torch = None

try:
    from ultralytics import YOLO
    _YOLO_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    YOLO = None
    _YOLO_IMPORT_ERROR = exc


logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self, config: dict[str, Any]) -> None:
        model_cfg = config['model']
        self.weights_path = Path(model_cfg['weights'])
        self.pretrained = model_cfg.get('pretrained', 'yolo11n-seg.pt')
        self.imgsz = int(model_cfg.get('imgsz', 640))
        self.conf = float(model_cfg.get('conf', 0.25))
        self.iou = float(model_cfg.get('iou', 0.5))
        self.max_det = int(model_cfg.get('max_det', 200))
        self.use_gpu = bool(model_cfg.get('use_gpu', True))
        self.enable_ensemble = bool(model_cfg.get('enable_ensemble', False))
        self.ensemble_weights = [Path(p) for p in model_cfg.get('ensemble_weights', [])]

        gpu_available = False
        if torch is not None:
            cuda_mod = getattr(torch, 'cuda', None)
            is_available = getattr(cuda_mod, 'is_available', None)
            if callable(is_available):
                try:
                    gpu_available = bool(is_available())
                except Exception:
                    gpu_available = False
        self.device = 'cuda:0' if self.use_gpu and gpu_available else 'cpu'
        self.model = None
        self.ensemble_models: list[Any] = []

    async def load(self) -> None:
        if YOLO is None:
            logger.error(
                'Ultralytics is not available, model cannot be loaded. '
                'Import error: %s',
                _YOLO_IMPORT_ERROR,
            )
            return

        def _load() -> Any:
            if self.weights_path.exists():
                logger.info('Loading model from weights: %s', self.weights_path)
                return YOLO(str(self.weights_path))

            logger.warning(
                'Weights %s not found, falling back to pretrained %s',
                self.weights_path,
                self.pretrained,
            )
            return YOLO(self.pretrained)

        self.model = await asyncio.to_thread(_load)
        self.ensemble_models = []
        if self.enable_ensemble and YOLO is not None:
            for weight in self.ensemble_weights:
                try:
                    self.ensemble_models.append(await asyncio.to_thread(YOLO, str(weight)))
                except Exception as exc:
                    logger.warning('Failed to load ensemble model %s: %s', weight, exc)
        logger.info('Model loaded successfully on device: %s', self.device)

    def is_loaded(self) -> bool:
        return self.model is not None

    async def predict(
        self,
        image_path: str,
        class_colors: dict[str, tuple[int, int, int]],
        overlay_alpha: float,
        conf: float | None = None,
        iou: float | None = None,
        max_det: int | None = None,
        use_ensemble: bool | None = None,
    ) -> dict[str, Any]:
        if self.model is None:
            raise ModelNotLoadedError('Model is not loaded.')

        ensemble_mode = self.enable_ensemble if use_ensemble is None else bool(use_ensemble)
        if ensemble_mode and self.ensemble_models:
            return await asyncio.to_thread(
                run_ensemble_inference,
                [self.model] + self.ensemble_models,
                image_path,
                class_colors,
                self.conf if conf is None else conf,
                self.iou if iou is None else iou,
                self.imgsz,
                self.max_det if max_det is None else max_det,
                overlay_alpha,
                self.device,
            )

        return await asyncio.to_thread(
            run_yolo_inference,
            self.model,
            image_path,
            class_colors,
            self.conf if conf is None else conf,
            self.iou if iou is None else iou,
            self.imgsz,
            self.max_det if max_det is None else max_det,
            overlay_alpha,
            self.device,
        )

    async def export_onnx(self, out_dir: str = 'models') -> str:
        if self.model is None:
            raise ModelNotLoadedError('Model is not loaded.')

        def _export() -> str:
            exported = self.model.export(
                format='onnx',
                dynamic=True,
                simplify=True,
                imgsz=self.imgsz,
            )
            return str(exported)

        path = await asyncio.to_thread(_export)
        logger.info('ONNX model exported: %s', path)
        return path

    async def export_int8(self, out_dir: str = 'models') -> str:
        onnx_path = await self.export_onnx(out_dir=out_dir)

        def _quantize() -> str:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            src = Path(onnx_path)
            dst = src.with_name(f'{src.stem}_int8{src.suffix}')
            quantize_dynamic(
                model_input=str(src),
                model_output=str(dst),
                weight_type=QuantType.QInt8,
            )
            return str(dst)

        qpath = await asyncio.to_thread(_quantize)
        logger.info('INT8 ONNX exported: %s', qpath)
        return qpath
