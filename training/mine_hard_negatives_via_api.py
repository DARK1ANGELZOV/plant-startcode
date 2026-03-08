from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from api.main import app


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CLASS_NAME_BY_ID = {0: "root", 1: "stem", 2: "leaves"}


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def parse_gt_presence(label_path: Path) -> dict[int, int]:
    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    c: Counter = Counter()
    if not text:
        return {}
    for row in text.splitlines():
        parts = row.split()
        if not parts:
            continue
        try:
            cid = int(float(parts[0]))
        except ValueError:
            continue
        c[cid] += 1
    return dict(c)


def collect_candidates(data_root: Path, split: str) -> list[tuple[Path, Path, dict[int, int]]]:
    labels_dir = data_root / "labels" / split
    images_dir = data_root / "images" / split
    out = []
    if not labels_dir.exists() or not images_dir.exists():
        return out
    for lbl in sorted(labels_dir.glob("*.txt")):
        gt = parse_gt_presence(lbl)
        if not gt:
            continue
        img = find_image(images_dir, lbl.stem)
        if img is None:
            continue
        out.append((img, lbl, gt))
    return out


def ensure_dirs(root: Path) -> None:
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "val").mkdir(parents=True, exist_ok=True)


def copy_pair(img: Path, lbl: Path, out_root: Path, split: str, name: str) -> None:
    out_img = out_root / "images" / split / f"{name}{img.suffix.lower()}"
    out_lbl = out_root / "labels" / split / f"{name}.txt"
    shutil.copy2(img, out_img)
    shutil.copy2(lbl, out_lbl)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _normalize_trust_score(score: float) -> float:
    s = _safe_float(score, 0.0)
    if s > 1.0:
        s = s / 100.0
    if s < 0.0:
        return 0.0
    if s > 1.0:
        return 1.0
    return s


@dataclass
class ApiInference:
    present: dict[str, bool]
    conf: dict[str, float]
    mode: str
    measurement_reliable: bool
    trust_score: float
    logic_passed: bool


def infer_presence_via_api(client: TestClient, image_path: Path) -> ApiInference:
    with image_path.open("rb") as fh:
        resp = client.post(
            "/chat/analyze",
            data={
                "message": "Analyze plant structures.",
                "crop": "Wheat",
                "camera_id": "default",
                "source_type": "lab_camera",
            },
            files={"image": (image_path.name, fh, "image/jpeg")},
        )

    if resp.status_code != 200:
        return ApiInference(
            present={"root": False, "stem": False, "leaves": False},
            conf={},
            mode="api_error",
            measurement_reliable=False,
            trust_score=0.0,
            logic_passed=False,
        )

    payload = resp.json()
    summary = (payload.get("result") or {}).get("summary") or {}
    seg = summary.get("segmentation") or {}
    conf = summary.get("confidence_by_class") or {}

    return ApiInference(
        present={
            "root": bool((seg.get("root") or {}).get("detected", False)),
            "stem": bool((seg.get("stem") or {}).get("detected", False)),
            "leaves": bool((seg.get("leaves") or {}).get("detected", False)),
        },
        conf={k: _safe_float(v, 0.0) for k, v in conf.items()},
        mode=str(summary.get("inference_mode", "unknown")),
        measurement_reliable=bool(summary.get("measurements_reliable", False)),
        trust_score=_safe_float(summary.get("measurement_trust_score"), 0.0),
        logic_passed=bool((summary.get("logic_checks") or {}).get("passed", False)),
    )


def _build_reasons(
    gt: dict[int, int],
    infer: ApiInference,
    min_class_conf: float,
    min_trust: float,
    weak_modes: set[str],
) -> list[str]:
    reasons: list[str] = []

    has_root_gt = gt.get(0, 0) > 0
    has_stem_gt = gt.get(1, 0) > 0
    root_conf = _safe_float(infer.conf.get("root"), 0.0)
    stem_conf = _safe_float(infer.conf.get("stem"), 0.0)

    if has_root_gt and not infer.present.get("root", False):
        reasons.append("miss_root")
    if has_stem_gt and not infer.present.get("stem", False):
        reasons.append("miss_stem")

    if has_root_gt and root_conf < min_class_conf:
        reasons.append("lowconf_root")
    if has_stem_gt and stem_conf < min_class_conf:
        reasons.append("lowconf_stem")

    if infer.mode in weak_modes:
        reasons.append(f"weak_mode:{infer.mode}")
    if not infer.measurement_reliable:
        reasons.append("unreliable_measurements")
    if _normalize_trust_score(infer.trust_score) < min_trust:
        reasons.append("low_trust")
    if not infer.logic_passed:
        reasons.append("logic_failed")

    uniq = []
    seen = set()
    for r in reasons:
        if r in seen:
            continue
        seen.add(r)
        uniq.append(r)
    return uniq


def _difficulty_score(reasons: list[str], infer: ApiInference, gt: dict[int, int], min_class_conf: float) -> float:
    score = 0.0

    for r in reasons:
        if r.startswith("miss_"):
            score += 2.0
        elif r.startswith("lowconf_"):
            score += 1.3
        elif r.startswith("weak_mode:"):
            score += 1.2
        elif r in {"unreliable_measurements", "logic_failed"}:
            score += 1.1
        elif r == "low_trust":
            score += 0.9

    if gt.get(0, 0) > 0:
        score += max(0.0, min_class_conf - _safe_float(infer.conf.get("root"), 0.0)) * 2.0
    if gt.get(1, 0) > 0:
        score += max(0.0, min_class_conf - _safe_float(infer.conf.get("stem"), 0.0)) * 2.0

    trust01 = _normalize_trust_score(infer.trust_score)
    score += max(0.0, 0.8 - trust01)
    return round(score, 5)


def _copy_golden_candidates(rows: list[dict[str, Any]], golden_root: Path, target: int) -> dict[str, int]:
    if golden_root.exists():
        shutil.rmtree(golden_root)
    ensure_dirs(golden_root)

    selected = rows[: max(0, int(target))]
    split_counts = {"train": 0, "val": 0}

    for row in selected:
        split = str(row.get("split", "train"))
        if split not in split_counts:
            split = "train"
        idx = split_counts[split]
        img = Path(str(row["image"]))
        lbl = Path(str(row["label"]))
        name = f"goldcand_{split}_{idx:06d}"
        copy_pair(img, lbl, golden_root, split, name)
        row["golden_name"] = name
        split_counts[split] += 1

    ds_yaml = golden_root / "dataset.yaml"
    ds_yaml.write_text(
        "\n".join(
            [
                f"path: {golden_root.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: root",
                "  1: stem",
                "  2: leaves",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return split_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine hard negatives for root/stem via API pipeline.")
    parser.add_argument("--data-root", type=str, default="data/hf_multisource_mega10")
    parser.add_argument("--out", type=str, default="data/hard_mined_rootstem_api")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=260)
    parser.add_argument("--val-limit", type=int, default=90)
    parser.add_argument("--max-hard-per-split", type=int, default=500)
    parser.add_argument("--min-class-conf", type=float, default=0.55)
    parser.add_argument("--min-trust", type=float, default=0.72)
    parser.add_argument(
        "--weak-modes",
        type=str,
        default="api_error,model_unavailable,model_low_confidence,model_no_detections,model_assisted_no_detections,heuristic_fallback",
    )
    parser.add_argument("--golden-out", type=str, default="data/golden_candidates_rootstem")
    parser.add_argument("--golden-target", type=int, default=400)
    args = parser.parse_args()

    random.seed(args.seed)

    out_root = Path(args.out)
    if out_root.exists():
        shutil.rmtree(out_root)
    ensure_dirs(out_root)

    weak_modes = {m.strip() for m in str(args.weak_modes).split(",") if m.strip()}

    hard_by_reason: Counter = Counter()
    hard_total = {"train": 0, "val": 0}
    details: dict[str, Any] = {}
    scored_rows: list[dict[str, Any]] = []

    with TestClient(app) as client:
        for split, lim in (("train", args.train_limit), ("val", args.val_limit)):
            rows = collect_candidates(Path(args.data_root), split)
            if len(rows) > lim:
                rows = random.sample(rows, lim)

            split_rows: list[dict[str, Any]] = []
            for img, lbl, gt in rows:
                infer = infer_presence_via_api(client, img)
                reasons = _build_reasons(
                    gt=gt,
                    infer=infer,
                    min_class_conf=float(args.min_class_conf),
                    min_trust=float(args.min_trust),
                    weak_modes=weak_modes,
                )
                score = _difficulty_score(
                    reasons=reasons,
                    infer=infer,
                    gt=gt,
                    min_class_conf=float(args.min_class_conf),
                )

                row_meta = {
                    "split": split,
                    "image": str(img),
                    "label": str(lbl),
                    "reasons": reasons,
                    "score": score,
                    "mode": infer.mode,
                    "measurement_reliable": infer.measurement_reliable,
                    "trust_score": infer.trust_score,
                    "logic_passed": infer.logic_passed,
                    "gt": {CLASS_NAME_BY_ID.get(k, str(k)): v for k, v in gt.items()},
                    "pred_conf": infer.conf,
                    "present": infer.present,
                }
                scored_rows.append(row_meta)

                if not reasons:
                    continue
                if hard_total[split] >= int(args.max_hard_per_split):
                    continue

                idx = hard_total[split]
                hard_name = f"hardapi_{split}_{idx:06d}"
                copy_pair(img, lbl, out_root, split, hard_name)
                hard_total[split] += 1
                for reason in reasons:
                    hard_by_reason[reason] += 1

                split_rows.append(
                    {
                        "name": hard_name,
                        "image": str(img),
                        "reasons": reasons,
                        "score": score,
                        "mode": infer.mode,
                        "measurement_reliable": infer.measurement_reliable,
                        "trust_score": infer.trust_score,
                        "logic_passed": infer.logic_passed,
                        "present": infer.present,
                        "gt": row_meta["gt"],
                        "pred_conf": infer.conf,
                    }
                )

            details[split] = {
                "checked": len(rows),
                "hard_found": len(split_rows),
                "rows": split_rows[:300],
            }

    ds_yaml = out_root / "dataset.yaml"
    ds_yaml.write_text(
        "\n".join(
            [
                f"path: {out_root.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: root",
                "  1: stem",
                "  2: leaves",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    scored_rows.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    golden_root = Path(args.golden_out)
    golden_counts = _copy_golden_candidates(
        rows=scored_rows,
        golden_root=golden_root,
        target=int(args.golden_target),
    )
    golden_manifest = golden_root / "golden_candidates_manifest.json"
    golden_manifest.write_text(
        json.dumps(
            {
                "target": int(args.golden_target),
                "selected_total": int(sum(golden_counts.values())),
                "selected_by_split": golden_counts,
                "top_candidates": scored_rows[: min(800, len(scored_rows))],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report = {
        "data_root": args.data_root,
        "thresholds": {
            "min_class_conf": float(args.min_class_conf),
            "min_trust": float(args.min_trust),
            "weak_modes": sorted(list(weak_modes)),
            "max_hard_per_split": int(args.max_hard_per_split),
        },
        "hard_total": hard_total,
        "hard_by_reason": dict(hard_by_reason),
        "dataset_yaml": str(ds_yaml),
        "golden_candidates": {
            "root": str(golden_root),
            "selected_by_split": golden_counts,
            "selected_total": int(sum(golden_counts.values())),
            "manifest": str(golden_manifest),
        },
        "details": details,
    }
    report_path = out_root / "mining_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
