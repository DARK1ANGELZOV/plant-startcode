from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def style_randomization(images: torch.Tensor, strength: float = 0.35) -> torch.Tensor:
    """Randomize channel-wise style statistics for domain generalization."""
    if images.ndim != 4:
        raise ValueError('Expected BCHW tensor.')
    b, c, _, _ = images.shape
    perm = torch.randperm(b, device=images.device)
    src = images
    tgt = images[perm]

    src_mean = src.mean(dim=(2, 3), keepdim=True)
    src_std = src.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    tgt_mean = tgt.mean(dim=(2, 3), keepdim=True)
    tgt_std = tgt.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)

    normalized = (src - src_mean) / src_std
    mixed = normalized * (src_std * (1.0 - strength) + tgt_std * strength) + (
        src_mean * (1.0 - strength) + tgt_mean * strength
    )
    return mixed.clamp(0.0, 1.0)


def coral_loss(source_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
    """CORAL feature alignment loss."""
    if source_feat.ndim > 2:
        source_feat = source_feat.flatten(2).mean(dim=2)
    if target_feat.ndim > 2:
        target_feat = target_feat.flatten(2).mean(dim=2)

    source_feat = source_feat - source_feat.mean(dim=0, keepdim=True)
    target_feat = target_feat - target_feat.mean(dim=0, keepdim=True)

    ns = max(1, source_feat.shape[0] - 1)
    nt = max(1, target_feat.shape[0] - 1)
    cs = (source_feat.t() @ source_feat) / ns
    ct = (target_feat.t() @ target_feat) / nt
    return torch.mean((cs - ct) ** 2)


def entropy_minimization_loss(logits: torch.Tensor) -> torch.Tensor:
    """Test-time adaptation objective: minimize prediction entropy."""
    probs = torch.softmax(logits, dim=1).clamp_min(1e-8)
    entropy = -(probs * torch.log(probs)).sum(dim=1)
    return entropy.mean()


def compute_feature_alignment_loss(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    weight: float = 0.05,
) -> torch.Tensor:
    return coral_loss(source_features, target_features) * float(weight)


def main() -> None:
    parser = argparse.ArgumentParser(description='Domain adaptation utilities smoke benchmark.')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--channels', type=int, default=3)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--output', default='reports/domain_adaptation_smoke.json')
    args = parser.parse_args()

    src = torch.rand(args.batch, args.channels, args.height, args.width)
    tgt = style_randomization(src, strength=0.4)

    src_feat = torch.rand(args.batch, 64, args.height // 8, args.width // 8)
    tgt_feat = torch.rand(args.batch, 64, args.height // 8, args.width // 8)
    align = compute_feature_alignment_loss(src_feat, tgt_feat, weight=0.1)

    logits = torch.randn(args.batch, 4, args.height // 4, args.width // 4)
    entropy = entropy_minimization_loss(logits)

    payload: dict[str, Any] = {
        'style_randomization_delta_mean': float((src - tgt).abs().mean().item()),
        'feature_alignment_loss': float(align.item()),
        'entropy_minimization_loss': float(entropy.item()),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding='utf-8')
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
