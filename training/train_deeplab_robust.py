from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

from training.robust_augmentations import build_train_augmentations, build_val_augmentations
from training.robust_corruptions import get_corruption_fn
from training.robust_dataset import MultiDomainSegDataset, create_domain_sampler, load_manifest_rows
from training.robust_metrics import evaluate_torch_segmentation, robustness_score
from utils.config import load_yaml
from utils.seed import set_global_seed


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg != 'auto':
        return torch.device(device_cfg)
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


def build_deeplab(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
    model = deeplabv3_resnet50(weights=weights, aux_loss=False)
    in_ch = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    grad_accum_steps: int,
) -> float:
    model.train()
    running = 0.0
    steps = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(dataloader, start=1):
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)

        with autocast(enabled=amp):
            logits = model(images)['out']
            loss = criterion(logits, masks) / max(1, grad_accum_steps)

        scaler.scale(loss).backward()

        if step % max(1, grad_accum_steps) == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running += float(loss.item()) * max(1, grad_accum_steps)
        steps += 1

    if steps == 0:
        return 0.0
    return running / steps


def run_stage(
    stage_name: str,
    model: nn.Module,
    train_rows,
    val_loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    writer: SummaryWriter | None,
    global_epoch_offset: int,
    adverse_ratio_start: float,
    adverse_ratio_end: float,
    output_dir: Path,
) -> tuple[nn.Module, dict[str, Any], int]:
    tr_cfg = cfg['training']
    hw_cfg = cfg['hardware']
    aug_cfg = cfg['augmentations']
    val_cfg = cfg['validation']

    epochs = int(tr_cfg['epochs'][stage_name])
    batch_size = int(tr_cfg['batch_size'])
    image_size = int(tr_cfg['image_size'])
    num_workers = int(hw_cfg.get('num_workers', 4))
    amp = bool(hw_cfg.get('amp', True) and device.type == 'cuda')
    grad_accum_steps = int(tr_cfg.get('grad_accum_steps', 1))

    optimizer = AdamW(model.parameters(), lr=float(tr_cfg['lr']), weight_decay=float(tr_cfg['weight_decay']))
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    scaler = GradScaler(enabled=amp)

    best = {'miou': -1.0, 'epoch': -1, 'path': ''}
    patience = int(tr_cfg.get('early_stopping_patience', 8))
    stale = 0

    model.to(device)

    for e in range(epochs):
        progress = (e + 1) / max(1, epochs)
        adverse_ratio = adverse_ratio_start + (adverse_ratio_end - adverse_ratio_start) * progress
        strength = cfg['curriculum']['corruption_strength_start'] + (
            cfg['curriculum']['corruption_strength_end'] - cfg['curriculum']['corruption_strength_start']
        ) * progress

        train_ds = MultiDomainSegDataset(
            train_rows,
            transform=build_train_augmentations(image_size=image_size, strength=strength, cfg=aug_cfg),
            mixup_prob=float(aug_cfg.get('mixup_prob', 0.0)),
            cutmix_prob=float(aug_cfg.get('cutmix_prob', 0.0)),
        )
        sampler = create_domain_sampler(train_rows, adverse_ratio=adverse_ratio)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=(device.type == 'cuda'),
            drop_last=False,
        )

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            device=device,
            amp=amp,
            grad_accum_steps=grad_accum_steps,
        )

        clean_metrics = evaluate_torch_segmentation(
            model=model,
            dataloader=val_loader,
            device=device,
            class_names=cfg['classes'],
            num_classes=len(cfg['classes']),
            boundary_dilation=int(val_cfg.get('boundary_dilation', 3)),
        )

        epoch_id = global_epoch_offset + e + 1
        if writer is not None:
            writer.add_scalar(f'{stage_name}/train_loss', train_loss, epoch_id)
            writer.add_scalar(f'{stage_name}/val_miou', clean_metrics.miou, epoch_id)
            writer.add_scalar(f'{stage_name}/val_boundary_iou', clean_metrics.boundary_iou, epoch_id)
            writer.add_scalar(f'{stage_name}/adverse_ratio', adverse_ratio, epoch_id)
            writer.add_scalar(f'{stage_name}/aug_strength', strength, epoch_id)

        if clean_metrics.miou > best['miou']:
            stale = 0
            best['miou'] = clean_metrics.miou
            best['epoch'] = epoch_id
            ckpt_path = output_dir / f'deeplab_{stage_name}_best.pt'
            torch.save(
                {
                    'model_state': model.state_dict(),
                    'epoch': epoch_id,
                    'stage': stage_name,
                    'metrics': {
                        'miou': clean_metrics.miou,
                        'mdice': clean_metrics.mdice,
                        'precision': clean_metrics.precision,
                        'recall': clean_metrics.recall,
                        'boundary_iou': clean_metrics.boundary_iou,
                    },
                },
                ckpt_path,
            )
            best['path'] = str(ckpt_path)
        else:
            stale += 1
            if stale >= patience:
                break

    summary = {
        'stage': stage_name,
        'best_miou': best['miou'],
        'best_epoch': best['epoch'],
        'best_checkpoint': best['path'],
    }
    return model, summary, global_epoch_offset + epochs


def main() -> None:
    parser = argparse.ArgumentParser(description='Robust DeepLabV3 training with curriculum and multi-domain mixing.')
    parser.add_argument('--config', default='configs/robust_train.yaml')
    parser.add_argument('--train-manifest', default='data/robust/train_manifest.jsonl')
    parser.add_argument('--val-manifest', default='data/robust/val_manifest.jsonl')
    parser.add_argument('--output', default='models/robust')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_global_seed(int(cfg.get('seed', 42)))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(str(cfg.get('hardware', {}).get('device', 'auto')))

    classes = cfg.get('classes', ['background', 'root', 'stem', 'leaves'])
    num_classes = len(classes)

    arch_cfg = cfg.get('architectures', {}).get('deeplabv3', {})
    model = build_deeplab(num_classes=num_classes, pretrained=bool(arch_cfg.get('pretrained', True)))

    writer = None
    if bool(cfg.get('logging', {}).get('tensorboard', True)):
        writer = SummaryWriter(log_dir=str(Path(cfg.get('project_root', 'runs/robust')) / 'tensorboard_deeplab'))

    train_all = load_manifest_rows(args.train_manifest, split='train')
    val_rows = load_manifest_rows(args.val_manifest, split='val')

    if not train_all:
        raise RuntimeError('Train manifest is empty. Build manifest before training.')
    if not val_rows:
        raise RuntimeError('Validation manifest is empty. Build val manifest before training.')

    val_ds = MultiDomainSegDataset(
        val_rows,
        transform=build_val_augmentations(int(cfg['training']['image_size'])),
        mixup_prob=0.0,
        cutmix_prob=0.0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg['training']['batch_size']),
        shuffle=False,
        num_workers=int(cfg['hardware'].get('num_workers', 4)),
        pin_memory=(device.type == 'cuda'),
    )

    stages = [
        ('pretrain', {'pretrain', 'multidomain'}, 0.0, 0.15),
        ('plant_finetune', {'plant'}, 0.0, 0.2),
        ('adverse_finetune', {'plant', 'adverse', 'multidomain'}, 0.2, float(cfg['curriculum']['adverse_max_ratio'])),
    ]

    stage_results = []
    epoch_offset = 0
    for stage_name, usages, adv_start, adv_end in stages:
        stage_rows = [r for r in train_all if r.usage in usages]
        if not stage_rows:
            stage_results.append({'stage': stage_name, 'skipped': True, 'reason': 'No samples for stage usages'})
            continue

        model, stage_summary, epoch_offset = run_stage(
            stage_name=stage_name,
            model=model,
            train_rows=stage_rows,
            val_loader=val_loader,
            cfg=cfg,
            device=device,
            writer=writer,
            global_epoch_offset=epoch_offset,
            adverse_ratio_start=adv_start,
            adverse_ratio_end=adv_end,
            output_dir=output_dir,
        )
        stage_results.append(stage_summary)

    clean = evaluate_torch_segmentation(
        model=model,
        dataloader=val_loader,
        device=device,
        class_names=classes,
        num_classes=num_classes,
        boundary_dilation=int(cfg['validation'].get('boundary_dilation', 3)),
    )

    corrupted = {}
    for corruption_name in cfg['validation'].get('corruption_tests', []):
        fn = get_corruption_fn(str(corruption_name))
        corrupted[corruption_name] = evaluate_torch_segmentation(
            model=model,
            dataloader=val_loader,
            device=device,
            class_names=classes,
            num_classes=num_classes,
            corruption_fn=fn,
            boundary_dilation=int(cfg['validation'].get('boundary_dilation', 3)),
        )

    robust = robustness_score(clean, corrupted)

    final_ckpt = output_dir / 'deeplab_robust_final.pt'
    torch.save({'model_state': model.state_dict(), 'classes': classes}, final_ckpt)

    report = {
        'device': str(device),
        'stages': stage_results,
        'clean': {
            'miou': clean.miou,
            'dice': clean.mdice,
            'precision': clean.precision,
            'recall': clean.recall,
            'boundary_iou': clean.boundary_iou,
            'per_class_iou': clean.per_class_iou,
        },
        'corrupted': {
            name: {
                'miou': m.miou,
                'dice': m.mdice,
                'precision': m.precision,
                'recall': m.recall,
                'boundary_iou': m.boundary_iou,
                'per_class_iou': m.per_class_iou,
            }
            for name, m in corrupted.items()
        },
        'robustness': robust,
        'checkpoint': str(final_ckpt.resolve()),
    }

    report_path = output_dir / 'deeplab_metrics.json'
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding='utf-8')

    if writer is not None:
        writer.close()

    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
