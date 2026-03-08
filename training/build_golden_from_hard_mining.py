from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_CATEGORY_RATIOS = {
    "dual_miss": 0.45,
    "root_only": 0.25,
    "stem_only": 0.25,
    "other": 0.05,
}


@dataclass
class Candidate:
    split: str
    name: str
    image_path: Path
    label_path: Path
    reasons: list[str]
    score: float
    category: str
    meta: dict[str, Any]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _normalize_path(value: str | Path) -> Path:
    return Path(str(value))


def _find_by_name(root: Path, split: str, name: str, labels: bool) -> Path | None:
    base_dir = root / ("labels" if labels else "images") / split
    if not base_dir.exists():
        return None
    if labels:
        p = base_dir / f"{name}.txt"
        return p if p.exists() else None
    for ext in IMAGE_EXTS:
        p = base_dir / f"{name}{ext}"
        if p.exists():
            return p
    return None


def _row_category(reasons: list[str]) -> str:
    has_root = "miss_root" in reasons
    has_stem = "miss_stem" in reasons
    if has_root and has_stem:
        return "dual_miss"
    if has_root:
        return "root_only"
    if has_stem:
        return "stem_only"
    return "other"


def _score_row(row: dict[str, Any]) -> float:
    reasons = [str(r) for r in (row.get("reasons") or [])]
    gt = row.get("gt") or {}
    pred_conf = row.get("pred_conf") or {}

    score = 0.0
    if "miss_root" in reasons:
        score += 3.0
    if "miss_stem" in reasons:
        score += 2.6
    if "miss_root" in reasons and "miss_stem" in reasons:
        score += 1.0

    root_conf = _safe_float(pred_conf.get("root"), 0.0)
    stem_conf = _safe_float(pred_conf.get("stem"), 0.0)
    score += max(0.0, 0.8 - root_conf) * 1.2
    score += max(0.0, 0.8 - stem_conf) * 1.0

    gt_root = int(_safe_float(gt.get("root"), 0.0))
    gt_stem = int(_safe_float(gt.get("stem"), 0.0))
    score += min(10, gt_root + gt_stem) * 0.08
    return round(score, 6)


def _load_candidates(report_path: Path, hard_root: Path) -> list[Candidate]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    details = payload.get("details") or {}

    out: list[Candidate] = []
    dedup_keys: set[tuple[str, str]] = set()

    for split in ("train", "val"):
        split_rows = ((details.get(split) or {}).get("rows") or [])
        for row in split_rows:
            name = str(row.get("name") or "").strip()
            if not name:
                continue

            row_image = _normalize_path(row.get("image", ""))
            row_label = _normalize_path(row.get("label", ""))

            image_path = row_image if row_image.exists() else _find_by_name(hard_root, split, name, labels=False)
            label_path = row_label if row_label.exists() else _find_by_name(hard_root, split, name, labels=True)
            if image_path is None or label_path is None:
                continue

            key = (split, name)
            if key in dedup_keys:
                continue
            dedup_keys.add(key)

            reasons = [str(r) for r in (row.get("reasons") or [])]
            category = _row_category(reasons)
            score = _score_row(row)

            out.append(
                Candidate(
                    split=split,
                    name=name,
                    image_path=image_path,
                    label_path=label_path,
                    reasons=reasons,
                    score=score,
                    category=category,
                    meta={
                        "gt": row.get("gt") or {},
                        "pred_conf": row.get("pred_conf") or {},
                    },
                )
            )
    return out


def _select_candidates(candidates: list[Candidate], target: int) -> list[Candidate]:
    by_category: dict[str, list[Candidate]] = defaultdict(list)
    for cand in candidates:
        by_category[cand.category].append(cand)

    for cat in by_category:
        by_category[cat].sort(key=lambda c: c.score, reverse=True)

    selected: list[Candidate] = []
    selected_keys: set[tuple[str, str]] = set()

    for cat, ratio in DEFAULT_CATEGORY_RATIOS.items():
        quota = int(round(target * ratio))
        pool = by_category.get(cat, [])
        take = min(quota, len(pool))
        for cand in pool[:take]:
            key = (cand.split, cand.name)
            if key in selected_keys:
                continue
            selected.append(cand)
            selected_keys.add(key)

    if len(selected) < target:
        global_pool = sorted(candidates, key=lambda c: c.score, reverse=True)
        for cand in global_pool:
            key = (cand.split, cand.name)
            if key in selected_keys:
                continue
            selected.append(cand)
            selected_keys.add(key)
            if len(selected) >= target:
                break

    return selected[:target]


def _copy_selected(selected: list[Candidate], out_root: Path) -> tuple[dict[str, int], dict[str, int], list[dict[str, Any]]]:
    if out_root.exists():
        shutil.rmtree(out_root)

    (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    split_counts = {"train": 0, "val": 0}
    category_counts: Counter[str] = Counter()
    entries: list[dict[str, Any]] = []

    for cand in selected:
        split = cand.split if cand.split in {"train", "val"} else "train"
        idx = split_counts[split]
        golden_name = f"gold_{split}_{idx:06d}"

        out_img = out_root / "images" / split / f"{golden_name}{cand.image_path.suffix.lower()}"
        out_lbl = out_root / "labels" / split / f"{golden_name}.txt"
        shutil.copy2(cand.image_path, out_img)
        shutil.copy2(cand.label_path, out_lbl)

        split_counts[split] += 1
        category_counts[cand.category] += 1

        entries.append(
            {
                "split": split,
                "golden_name": golden_name,
                "image": str(out_img.as_posix()),
                "label": str(out_lbl.as_posix()),
                "source_image": str(cand.image_path.as_posix()),
                "source_label": str(cand.label_path.as_posix()),
                "source_name": cand.name,
                "score": cand.score,
                "category": cand.category,
                "reasons": cand.reasons,
                "gt": cand.meta.get("gt", {}),
                "pred_conf": cand.meta.get("pred_conf", {}),
                "review_status": "pending",
            }
        )

    return split_counts, dict(category_counts), entries


def _write_dataset_yaml(out_root: Path) -> Path:
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
    return ds_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 300-500 case golden candidate set from hard-mining report.")
    parser.add_argument("--report", type=str, default="data/hard_mined_rootstem_rs/mining_report.json")
    parser.add_argument("--hard-root", type=str, default="data/hard_mined_rootstem_rs")
    parser.add_argument("--out", type=str, default="data/golden_rootstem_400")
    parser.add_argument("--target", type=int, default=400)
    parser.add_argument("--min-target", type=int, default=300)
    parser.add_argument("--max-target", type=int, default=500)
    parser.add_argument("--strict-min", action="store_true")
    parser.add_argument("--report-out", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    target = int(args.target)
    min_target = int(args.min_target)
    max_target = int(args.max_target)
    if target < min_target or target > max_target:
        raise ValueError(f"target must be in [{min_target}, {max_target}], got {target}")

    report_path = Path(args.report)
    hard_root = Path(args.hard_root)
    out_root = Path(args.out)
    if not report_path.exists():
        raise FileNotFoundError(f"Mining report not found: {report_path}")
    if not hard_root.exists():
        raise FileNotFoundError(f"Hard-mined dataset root not found: {hard_root}")

    candidates = _load_candidates(report_path=report_path, hard_root=hard_root)
    if not candidates:
        raise RuntimeError("No candidates loaded from mining report.")

    selected = _select_candidates(candidates, target=target)
    if args.strict_min and len(selected) < min_target:
        raise RuntimeError(f"Selected {len(selected)} cases, required at least {min_target}.")

    split_counts, category_counts, entries = _copy_selected(selected, out_root=out_root)
    ds_yaml = _write_dataset_yaml(out_root)

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_report": str(report_path.as_posix()),
        "source_hard_root": str(hard_root.as_posix()),
        "target": target,
        "selected_total": len(entries),
        "selected_by_split": split_counts,
        "selected_by_category": category_counts,
        "dataset_yaml": str(ds_yaml.as_posix()),
        "entries": entries,
    }

    manifest_path = out_root / "golden_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path_out = Path(args.report_out) if args.report_out else (out_root / "golden_build_report.json")
    report_payload = {
        "target": target,
        "loaded_candidates": len(candidates),
        "selected_total": len(entries),
        "selected_by_split": split_counts,
        "selected_by_category": category_counts,
        "manifest": str(manifest_path.as_posix()),
        "dataset_yaml": str(ds_yaml.as_posix()),
    }
    report_path_out.parent.mkdir(parents=True, exist_ok=True)
    report_path_out.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

