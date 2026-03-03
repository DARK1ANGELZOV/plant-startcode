from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


IGNORE_INDEX = 255


@dataclass
class SampleRow:
    image_path: str
    mask_path: str
    domain: str
    usage: str
    split: str
    is_adverse: bool
    has_labels: bool


def load_manifest_rows(
    manifest_path: str,
    usages: set[str] | None = None,
    split: str | None = None,
) -> list[SampleRow]:
    rows: list[SampleRow] = []
    for raw in Path(manifest_path).read_text(encoding='utf-8').splitlines():
        raw = raw.strip()
        if not raw:
            continue
        item = json.loads(raw)
        row = SampleRow(**item)
        if split and row.split != split:
            continue
        if usages and row.usage not in usages:
            continue
        rows.append(row)
    return rows


def create_domain_sampler(
    rows: list[SampleRow],
    adverse_ratio: float,
) -> WeightedRandomSampler:
    weights = []
    adv = max(0.0, min(1.0, adverse_ratio))
    for r in rows:
        if r.is_adverse:
            weights.append(0.2 + 1.6 * adv)
        else:
            weights.append(1.0)

    w = torch.tensor(weights, dtype=torch.float)
    return WeightedRandomSampler(weights=w, num_samples=len(rows), replacement=True)


class MultiDomainSegDataset(Dataset):
    def __init__(
        self,
        rows: list[SampleRow],
        transform=None,
        mixup_prob: float = 0.0,
        cutmix_prob: float = 0.0,
    ) -> None:
        self.rows = rows
        self.transform = transform
        self.mixup_prob = float(mixup_prob)
        self.cutmix_prob = float(cutmix_prob)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_raw(self, idx: int) -> tuple[np.ndarray, np.ndarray, SampleRow]:
        row = self.rows[idx]
        img = cv2.imread(row.image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f'Image read failed: {row.image_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(row.mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f'Mask read failed: {row.mask_path}')
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = mask.astype(np.uint8)
        return img, mask, row

    def _apply_transform(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.transform is None:
            return image, mask
        out = self.transform(image=image, mask=mask)
        return out['image'], out['mask']

    @staticmethod
    def _to_tensor(image: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        image_t = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        mask_t = torch.from_numpy(mask.astype(np.int64))
        return image_t, mask_t

    def _mixup(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx2 = random.randint(0, len(self.rows) - 1)
        image2, mask2, _ = self._load_raw(idx2)
        image2, mask2 = self._apply_transform(image2, mask2)

        lam = float(np.random.beta(0.4, 0.4))
        mixed_img = (image.astype(np.float32) * lam + image2.astype(np.float32) * (1.0 - lam)).astype(np.uint8)

        chooser = np.random.rand(*mask.shape)
        mixed_mask = np.where(chooser < lam, mask, mask2).astype(np.uint8)
        return mixed_img, mixed_mask

    def _cutmix(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx2 = random.randint(0, len(self.rows) - 1)
        image2, mask2, _ = self._load_raw(idx2)
        image2, mask2 = self._apply_transform(image2, mask2)

        h, w = image.shape[:2]
        rw = int(w * random.uniform(0.25, 0.55))
        rh = int(h * random.uniform(0.25, 0.55))
        x1 = random.randint(0, max(0, w - rw))
        y1 = random.randint(0, max(0, h - rh))
        x2 = min(w, x1 + rw)
        y2 = min(h, y1 + rh)

        mixed_img = image.copy()
        mixed_mask = mask.copy()
        mixed_img[y1:y2, x1:x2] = image2[y1:y2, x1:x2]
        mixed_mask[y1:y2, x1:x2] = mask2[y1:y2, x1:x2]
        return mixed_img, mixed_mask

    def __getitem__(self, idx: int) -> dict:
        image, mask, row = self._load_raw(idx)
        image, mask = self._apply_transform(image, mask)

        if self.mixup_prob > 0 and random.random() < self.mixup_prob:
            image, mask = self._mixup(image, mask)

        if self.cutmix_prob > 0 and random.random() < self.cutmix_prob:
            image, mask = self._cutmix(image, mask)

        image_t, mask_t = self._to_tensor(image, mask)
        return {
            'image': image_t,
            'mask': mask_t,
            'domain': row.domain,
            'usage': row.usage,
            'is_adverse': row.is_adverse,
            'image_path': row.image_path,
        }
