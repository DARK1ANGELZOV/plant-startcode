from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch


@dataclass
class SegMetrics:
    miou: float
    mdice: float
    precision: float
    recall: float
    boundary_iou: float
    per_class_iou: dict[str, float]


def _confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    mask = target != ignore_index
    pred = pred[mask].view(-1)
    target = target[mask].view(-1)
    if pred.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.float64, device=pred.device)

    idx = target * num_classes + pred
    hist = torch.bincount(idx, minlength=num_classes * num_classes)
    return hist.reshape(num_classes, num_classes).to(torch.float64)


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d > 0 else 0.0


def _boundary_mask(mask: np.ndarray, dilation: int = 3) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = (mask > 0).astype(np.uint8)
    eroded = cv2.erode(mask_u8, kernel, iterations=dilation)
    boundary = (mask_u8 - eroded) > 0
    return boundary.astype(np.uint8)


def boundary_iou(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
    dilation: int = 3,
) -> float:
    vals = []
    valid = target != ignore_index

    for cls_id in range(1, num_classes):
        pred_cls = np.logical_and(pred == cls_id, valid)
        tgt_cls = np.logical_and(target == cls_id, valid)

        b_pred = _boundary_mask(pred_cls.astype(np.uint8), dilation=dilation)
        b_tgt = _boundary_mask(tgt_cls.astype(np.uint8), dilation=dilation)

        union = np.logical_or(b_pred > 0, b_tgt > 0).sum()
        inter = np.logical_and(b_pred > 0, b_tgt > 0).sum()
        vals.append(_safe_div(float(inter), float(union)))

    return float(np.mean(vals)) if vals else 0.0


def compute_seg_metrics(
    confmat: torch.Tensor,
    class_names: list[str],
    boundary_scores: list[float] | None = None,
) -> SegMetrics:
    num_classes = confmat.shape[0]

    ious = []
    dices = []
    precisions = []
    recalls = []
    per_class_iou: dict[str, float] = {}

    for c in range(1, num_classes):
        tp = float(confmat[c, c])
        fp = float(confmat[:, c].sum() - tp)
        fn = float(confmat[c, :].sum() - tp)

        iou = _safe_div(tp, tp + fp + fn)
        dice = _safe_div(2 * tp, 2 * tp + fp + fn)
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)

        label = class_names[c] if c < len(class_names) else f'class_{c}'
        per_class_iou[label] = iou

        ious.append(iou)
        dices.append(dice)
        precisions.append(prec)
        recalls.append(rec)

    return SegMetrics(
        miou=float(np.mean(ious)) if ious else 0.0,
        mdice=float(np.mean(dices)) if dices else 0.0,
        precision=float(np.mean(precisions)) if precisions else 0.0,
        recall=float(np.mean(recalls)) if recalls else 0.0,
        boundary_iou=float(np.mean(boundary_scores)) if boundary_scores else 0.0,
        per_class_iou=per_class_iou,
    )


def evaluate_torch_segmentation(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    class_names: list[str],
    num_classes: int,
    corruption_fn=None,
    boundary_dilation: int = 3,
) -> SegMetrics:
    model.eval()
    confmat = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)
    boundary_scores: list[float] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            targets = batch['mask'].to(device)

            if corruption_fn is not None:
                images = corruption_fn(images)

            out = model(images)
            logits = out['out'] if isinstance(out, dict) else out
            preds = torch.argmax(logits, dim=1)

            confmat += _confusion_matrix(preds, targets, num_classes=num_classes)

            preds_np = preds.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            for p, t in zip(preds_np, targets_np):
                boundary_scores.append(
                    boundary_iou(
                        p,
                        t,
                        num_classes=num_classes,
                        dilation=boundary_dilation,
                    )
                )

    return compute_seg_metrics(confmat, class_names=class_names, boundary_scores=boundary_scores)


def robustness_score(clean: SegMetrics, corrupted: dict[str, SegMetrics]) -> dict[str, Any]:
    drops = {}
    for key, val in corrupted.items():
        drops[key] = max(0.0, clean.miou - val.miou)
    mean_drop = float(np.mean(list(drops.values()))) if drops else 0.0

    # Higher is better.
    score = max(0.0, clean.miou * (1.0 - mean_drop))

    return {
        'clean_miou': clean.miou,
        'mean_drop': mean_drop,
        'drop_by_corruption': drops,
        'robustness_score': score,
    }
