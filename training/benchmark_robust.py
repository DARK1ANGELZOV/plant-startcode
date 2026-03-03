from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from utils.config import load_yaml

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import torch
except Exception:
    torch = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    arr = np.array(values, dtype=np.float64)
    return float(np.percentile(arr, q))


def _summary_ms(latencies_sec: list[float]) -> dict[str, float]:
    lat_ms = [v * 1000.0 for v in latencies_sec]
    return {
        'mean_ms': float(statistics.fmean(lat_ms)) if lat_ms else 0.0,
        'p50_ms': _percentile(lat_ms, 50),
        'p90_ms': _percentile(lat_ms, 90),
        'p95_ms': _percentile(lat_ms, 95),
        'max_ms': max(lat_ms) if lat_ms else 0.0,
        'min_ms': min(lat_ms) if lat_ms else 0.0,
    }


def _make_image_numpy(image_size: int) -> np.ndarray:
    return np.random.randint(0, 255, size=(image_size, image_size, 3), dtype=np.uint8)


def _make_tensor_batch(image_size: int, batch_size: int, device: str):
    if torch is None:
        raise RuntimeError('torch is not available.')
    t = torch.rand((batch_size, 3, image_size, image_size), dtype=torch.float32)
    return t.to(device)


def _sync_if_needed(device: str) -> None:
    if torch is not None and device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_yolo_pt(
    weights_path: str,
    image_size: int,
    iterations: int,
    warmup: int,
    device: str,
) -> dict[str, Any]:
    if YOLO is None:
        raise RuntimeError('ultralytics is not available.')
    model = YOLO(weights_path)
    image = _make_image_numpy(image_size)

    for _ in range(warmup):
        model.predict(source=image, imgsz=image_size, device=device, verbose=False, save=False)
    _sync_if_needed(device)

    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        model.predict(source=image, imgsz=image_size, device=device, verbose=False, save=False)
        _sync_if_needed(device)
        latencies.append(time.perf_counter() - start)

    return _summary_ms(latencies)


def benchmark_deeplab_pt(
    checkpoint_path: str,
    num_classes: int,
    image_size: int,
    iterations: int,
    warmup: int,
    device: str,
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError('torch is not available.')
    from training.train_deeplab_robust import build_deeplab

    model = build_deeplab(num_classes=num_classes, pretrained=False)
    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state['model_state'], strict=True)
    model.to(device)
    model.eval()

    with torch.no_grad():
        inputs = _make_tensor_batch(image_size=image_size, batch_size=1, device=device)
        for _ in range(warmup):
            _ = model(inputs)
        _sync_if_needed(device)

        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(inputs)
            _sync_if_needed(device)
            latencies.append(time.perf_counter() - start)

    return _summary_ms(latencies)


def benchmark_onnx(
    onnx_path: str,
    image_size: int,
    iterations: int,
    warmup: int,
    provider: str,
) -> dict[str, Any]:
    if ort is None:
        raise RuntimeError('onnxruntime is not available.')

    providers = [provider]
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    shape = input_info.shape
    dtype = np.float32

    batch = 1
    channels = 3
    height = image_size
    width = image_size

    if len(shape) == 4:
        if isinstance(shape[1], int) and shape[1] > 0:
            channels = shape[1]
        if isinstance(shape[2], int) and shape[2] > 0:
            height = shape[2]
        if isinstance(shape[3], int) and shape[3] > 0:
            width = shape[3]

    data = np.random.rand(batch, channels, height, width).astype(dtype)

    for _ in range(warmup):
        _ = session.run(None, {input_name: data})

    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = session.run(None, {input_name: data})
        latencies.append(time.perf_counter() - start)

    return _summary_ms(latencies)


def _device_list(cpu_only: bool) -> list[str]:
    devices = ['cpu']
    if not cpu_only and torch is not None and torch.cuda.is_available():
        devices.append('cuda:0')
    return devices


def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark robust segmentation models for production latency.')
    parser.add_argument('--config', default='configs/robust_train.yaml')
    parser.add_argument('--architecture', choices=['yolo_seg', 'deeplabv3', 'auto'], default='auto')
    parser.add_argument('--weights', default='')
    parser.add_argument('--onnx', default='')
    parser.add_argument('--image-size', type=int, default=1024)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--output', default='models/robust/benchmark.json')
    parser.add_argument('--target-ms', type=float, default=200.0)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    classes = cfg.get('classes', ['background', 'root', 'stem', 'leaves'])

    architecture = args.architecture
    if architecture == 'auto':
        best_path = Path('models/robust/best_architecture.json')
        if best_path.exists():
            payload = json.loads(best_path.read_text(encoding='utf-8'))
            architecture = payload.get('best_architecture', 'yolo_seg')
        else:
            architecture = 'yolo_seg'

    weights = args.weights
    onnx_path = args.onnx

    if not weights:
        if architecture == 'yolo_seg':
            weights = 'models/robust/yolo_robust_best.pt'
        else:
            weights = 'models/robust/deeplab_robust_final.pt'

    if not onnx_path and architecture == 'yolo_seg':
        onnx_path = 'models/robust/yolo_robust_best.onnx'

    report: dict[str, Any] = {
        'architecture': architecture,
        'image_size': args.image_size,
        'iterations': args.iterations,
        'warmup': args.warmup,
        'devices': {},
        'onnx': {},
    }

    for device in _device_list(cpu_only=args.cpu_only):
        if architecture == 'yolo_seg':
            metrics = benchmark_yolo_pt(
                weights_path=weights,
                image_size=args.image_size,
                iterations=args.iterations,
                warmup=args.warmup,
                device=device,
            )
        else:
            metrics = benchmark_deeplab_pt(
                checkpoint_path=weights,
                num_classes=len(classes),
                image_size=args.image_size,
                iterations=args.iterations,
                warmup=args.warmup,
                device=device,
            )
        metrics['meets_target_lt_200ms'] = metrics['p95_ms'] < args.target_ms
        report['devices'][device] = metrics

    onnx_file = Path(onnx_path)
    if onnx_file.exists() and ort is not None:
        providers = ['CPUExecutionProvider']
        if not args.cpu_only:
            available = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available:
                providers.insert(0, 'CUDAExecutionProvider')

        for provider in providers:
            onnx_metrics = benchmark_onnx(
                onnx_path=str(onnx_file),
                image_size=args.image_size,
                iterations=args.iterations,
                warmup=args.warmup,
                provider=provider,
            )
            onnx_metrics['meets_target_lt_200ms'] = onnx_metrics['p95_ms'] < args.target_ms
            report['onnx'][provider] = onnx_metrics

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
