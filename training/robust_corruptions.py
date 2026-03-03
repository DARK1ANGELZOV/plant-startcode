from __future__ import annotations

import cv2
import numpy as np
import torch


def _to_np(images: torch.Tensor) -> np.ndarray:
    arr = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def _to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(arr).float().permute(0, 3, 1, 2) / 255.0
    return t.to(device)


def gaussian_blur(images: torch.Tensor, ksize: int = 5) -> torch.Tensor:
    arr = _to_np(images)
    out = np.stack([cv2.GaussianBlur(img, (ksize, ksize), 0) for img in arr], axis=0)
    return _to_tensor(out, device=images.device)


def motion_blur(images: torch.Tensor, ksize: int = 9) -> torch.Tensor:
    arr = _to_np(images)
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0 / ksize
    out = np.stack([cv2.filter2D(img, -1, kernel) for img in arr], axis=0)
    return _to_tensor(out, device=images.device)


def additive_noise(images: torch.Tensor, std: float = 0.08) -> torch.Tensor:
    noise = torch.randn_like(images) * std
    return torch.clamp(images + noise, 0.0, 1.0)


def jpeg_artifacts(images: torch.Tensor, quality: int = 30) -> torch.Tensor:
    arr = _to_np(images)
    encoded = []
    for img in arr:
        ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            encoded.append(img)
            continue
        dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        encoded.append(dec if dec is not None else img)
    out = np.stack(encoded, axis=0)
    return _to_tensor(out, device=images.device)


def fog(images: torch.Tensor, strength: float = 0.35) -> torch.Tensor:
    arr = _to_np(images).astype(np.float32)
    h, w = arr.shape[1:3]
    y = np.linspace(-1, 1, h).reshape(h, 1)
    x = np.linspace(-1, 1, w).reshape(1, w)
    dist = np.sqrt(x * x + y * y)
    mask = np.clip(1.0 - dist, 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=15)
    mask = mask[None, :, :, None]
    fog_color = np.full_like(arr, 235.0)
    out = arr * (1.0 - strength * mask) + fog_color * (strength * mask)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return _to_tensor(out, device=images.device)


def rain(images: torch.Tensor, drops: int = 900, alpha: float = 0.2) -> torch.Tensor:
    arr = _to_np(images)
    b, h, w, _ = arr.shape
    out = arr.copy()
    for i in range(b):
        layer = np.zeros((h, w, 3), dtype=np.uint8)
        ys = np.random.randint(0, h, size=drops)
        xs = np.random.randint(0, w, size=drops)
        lengths = np.random.randint(6, 18, size=drops)
        for x, y, ln in zip(xs, ys, lengths):
            x2 = min(w - 1, x + 2)
            y2 = min(h - 1, y + int(ln))
            cv2.line(layer, (int(x), int(y)), (int(x2), int(y2)), (190, 190, 190), 1)
        out[i] = cv2.addWeighted(out[i], 1.0, layer, alpha, 0)
    return _to_tensor(out, device=images.device)


def get_corruption_fn(name: str):
    name = name.lower()
    if name == 'gaussian_blur':
        return lambda x: gaussian_blur(x, ksize=5)
    if name == 'motion_blur':
        return lambda x: motion_blur(x, ksize=9)
    if name == 'noise':
        return lambda x: additive_noise(x, std=0.08)
    if name == 'jpeg':
        return lambda x: jpeg_artifacts(x, quality=28)
    if name == 'fog':
        return lambda x: fog(x, strength=0.4)
    if name == 'rain':
        return lambda x: rain(x, drops=900, alpha=0.25)
    raise ValueError(f'Unknown corruption: {name}')
