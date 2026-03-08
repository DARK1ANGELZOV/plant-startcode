from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from utils.config import load_yaml

try:
    import torch
except Exception:
    torch = None


def _resolve_device(raw: Any) -> str | int:
    if raw not in (None, "", "auto"):
        return raw
    if torch is not None and bool(getattr(torch.cuda, "is_available", lambda: False)()):
        return 0
    return "cpu"


def _resolve_best(project: str, name: str) -> Path:
    p = Path(project) / name / "weights" / "best.pt"
    if p.exists():
        return p
    p2 = Path.cwd() / project / name / "weights" / "best.pt"
    if p2.exists():
        return p2
    # Ultralytics can also nest under runs/segment/runs/segment on Windows.
    p3 = Path("C:/Users/Dark_Angel/runs/segment/runs/segment") / name / "weights" / "best.pt"
    if p3.exists():
        return p3
    raise FileNotFoundError(f"best.pt not found for run {name}")


def _train_stage(model_path: str, cfg: dict[str, Any]) -> Path:
    model = YOLO(model_path)
    model.train(
        data=cfg["data"],
        epochs=int(cfg["epochs"]),
        batch=int(cfg["batch"]),
        imgsz=int(cfg["imgsz"]),
        device=_resolve_device(cfg.get("device", "cpu")),
        workers=int(cfg.get("workers", 0)),
        amp=bool(cfg.get("amp", True)),
        project=str(cfg.get("project", "runs/segment")),
        name=str(cfg["name"]),
        optimizer=str(cfg.get("optimizer", "AdamW")),
        lr0=float(cfg.get("lr0", 2e-4)),
        lrf=float(cfg.get("lrf", 1e-2)),
        warmup_epochs=float(cfg.get("warmup_epochs", 1.0)),
        patience=int(cfg.get("patience", 20)),
        mosaic=float(cfg.get("mosaic", 0.2)),
        mixup=float(cfg.get("mixup", 0.0)),
        copy_paste=float(cfg.get("copy_paste", 0.0)),
        close_mosaic=int(cfg.get("close_mosaic", 10)),
        hsv_h=float(cfg.get("hsv_h", 0.02)),
        hsv_s=float(cfg.get("hsv_s", 0.3)),
        hsv_v=float(cfg.get("hsv_v", 0.2)),
        fliplr=float(cfg.get("fliplr", 0.5)),
        flipud=float(cfg.get("flipud", 0.0)),
        overlap_mask=bool(cfg.get("overlap_mask", True)),
        mask_ratio=int(cfg.get("mask_ratio", 4)),
        box=float(cfg.get("box", 7.5)),
        cls=float(cfg.get("cls", 0.5)),
        dfl=float(cfg.get("dfl", 1.5)),
        cos_lr=bool(cfg.get("cos_lr", False)),
        dropout=float(cfg.get("dropout", 0.0)),
        fraction=float(cfg.get("fraction", 1.0)),
        cache=cfg.get("cache", False),
        seed=42,
        exist_ok=True,
    )
    return _resolve_best(str(cfg.get("project", "runs/segment")), str(cfg["name"]))


def _evaluate(model_path: str, data_yaml: str, imgsz: int, batch: int, workers: int) -> dict[str, Any]:
    model = YOLO(model_path)
    val = model.val(
        data=data_yaml,
        split="val",
        imgsz=int(imgsz),
        batch=int(batch),
        workers=int(workers),
        device="cpu",
        plots=False,
        save_json=False,
    )
    seg = getattr(val, "seg", None)
    maps = list(getattr(seg, "maps", [])) if seg is not None else []
    iou = {model.names.get(i, str(i)): float(v) for i, v in enumerate(maps)}
    return {
        "iou_per_class": iou,
        "precision": float(getattr(seg, "mp", 0.0)) if seg is not None else 0.0,
        "recall": float(getattr(seg, "mr", 0.0)) if seg is not None else 0.0,
        "mAP50": float(getattr(seg, "map50", 0.0)) if seg is not None else 0.0,
        "mAP50_95": float(getattr(seg, "map", 0.0)) if seg is not None else 0.0,
    }


def _safe_pct(before: float, after: float) -> float | None:
    if abs(before) < 1e-12:
        return None
    return (after - before) / before * 100.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ideal root/stem fine-tune cycle.")
    parser.add_argument("--config", type=str, default="configs/train_rootstem_ideal.yaml")
    parser.add_argument("--baseline", type=str, default="reports/eval_before_mega10.json")
    parser.add_argument("--out", type=str, default="reports/ideal_rootstem_cycle_summary.json")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))

    stage_a = cfg["stage_a"]
    stage_b = cfg.get("stage_b", {})
    eval_cfg = cfg.get("eval", {})

    best_a = _train_stage(stage_a["model"], stage_a)
    best_final = best_a
    if bool(stage_b.get("enabled", True)):
        stage_b_cfg = dict(stage_b)
        if "model" not in stage_b_cfg:
            stage_b_cfg["model"] = str(best_a)
        best_b = _train_stage(str(best_a), stage_b_cfg)
        best_final = best_b

    metrics = _evaluate(
        model_path=str(best_final),
        data_yaml=str(eval_cfg.get("benchmark_data", "data/hf_multisource_mega10/dataset.yaml")),
        imgsz=int(eval_cfg.get("imgsz", 512)),
        batch=int(eval_cfg.get("batch", 1)),
        workers=int(eval_cfg.get("workers", 0)),
    )

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_final, models_dir / "best_max.pt")
    model = YOLO(str(models_dir / "best_max.pt"))
    exported = model.export(format="onnx", opset=13, simplify=True, imgsz=int(eval_cfg.get("imgsz", 512)))
    exported_path = Path(str(exported))
    if exported_path.exists():
        shutil.copy2(exported_path, models_dir / "best_max.onnx")

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "config": args.config,
        "best_stage_a": str(best_a),
        "best_final": str(best_final),
        "metrics_before": baseline,
        "metrics_after": metrics,
        "delta_percent": {
            "precision": _safe_pct(float(baseline.get("precision", 0.0)), float(metrics.get("precision", 0.0))),
            "recall": _safe_pct(float(baseline.get("recall", 0.0)), float(metrics.get("recall", 0.0))),
            "mAP50": _safe_pct(float(baseline.get("mAP50", 0.0)), float(metrics.get("mAP50", 0.0))),
            "mAP50_95": _safe_pct(float(baseline.get("mAP50_95", 0.0)), float(metrics.get("mAP50_95", 0.0))),
            "leaves_iou": _safe_pct(
                float((baseline.get("iou_per_class") or {}).get("leaves", 0.0)),
                float((metrics.get("iou_per_class") or {}).get("leaves", 0.0)),
            ),
        },
        "artifacts": {
            "model": "models/best_max.pt",
            "onnx": "models/best_max.onnx",
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
