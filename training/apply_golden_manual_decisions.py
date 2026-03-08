from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path


APPROVED_STATUSES = {"approved", "approve", "ok", "accept", "accepted", "yes", "1"}
REJECTED_STATUSES = {"rejected", "reject", "no", "0"}
PENDING_STATUSES = {"pending", "skip", "unsure", ""}


def _read_review_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return [dict(r) for r in csv.DictReader(fp)]


def _normalize_status(value: str) -> str:
    v = str(value or "").strip().lower()
    if v in APPROVED_STATUSES:
        return "approved"
    if v in REJECTED_STATUSES:
        return "rejected"
    if v in PENDING_STATUSES:
        return "pending"
    return "pending"


def _write_dataset_yaml(out_root: Path) -> Path:
    p = out_root / "dataset.yaml"
    p.write_text(
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
    return p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build validated golden dataset from manual review CSV decisions.")
    parser.add_argument("--review-csv", type=str, default="data/golden_rootstem_400_review/manual_review.csv")
    parser.add_argument("--source-root", type=str, default="data/golden_rootstem_400")
    parser.add_argument("--out", type=str, default="data/golden_rootstem_400_validated")
    parser.add_argument("--allow-pending-as-approved", action="store_true")
    parser.add_argument("--min-approved", type=int, default=300)
    return parser.parse_args()


def _resolve_path(raw: str, source_root: Path, split: str, name: str, label: bool) -> Path | None:
    p = Path(str(raw))
    if p.exists():
        return p
    if not name:
        return None
    base = source_root / ("labels" if label else "images") / split
    if label:
        cand = base / f"{name}.txt"
        return cand if cand.exists() else None
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        cand = base / f"{name}{ext}"
        if cand.exists():
            return cand
    return None


def main() -> None:
    args = parse_args()
    review_csv = Path(args.review_csv)
    source_root = Path(args.source_root)
    out_root = Path(args.out)
    if not review_csv.exists():
        raise FileNotFoundError(f"Review CSV not found: {review_csv}")
    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    rows = _read_review_csv(review_csv)
    if out_root.exists():
        shutil.rmtree(out_root)
    (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    counts = Counter()
    accepted_entries: list[dict[str, str]] = []
    copied_by_split = {"train": 0, "val": 0}

    for row in rows:
        split = str(row.get("split", "train")).strip().lower()
        if split not in {"train", "val"}:
            split = "train"
        name = str(row.get("golden_name", "")).strip()
        status = _normalize_status(row.get("status", ""))

        if status == "pending" and args.allow_pending_as_approved:
            status = "approved"

        counts[status] += 1
        if status != "approved":
            continue

        img = _resolve_path(row.get("image_path", ""), source_root, split=split, name=name, label=False)
        lbl = _resolve_path(row.get("label_path", ""), source_root, split=split, name=name, label=True)
        if img is None or lbl is None:
            counts["missing_files"] += 1
            continue

        idx = copied_by_split[split]
        out_name = f"goldval_{split}_{idx:06d}"
        out_img = out_root / "images" / split / f"{out_name}{img.suffix.lower()}"
        out_lbl = out_root / "labels" / split / f"{out_name}.txt"
        shutil.copy2(img, out_img)
        shutil.copy2(lbl, out_lbl)
        copied_by_split[split] += 1

        accepted_entries.append(
            {
                "split": split,
                "validated_name": out_name,
                "source_name": name,
                "image": out_img.as_posix(),
                "label": out_lbl.as_posix(),
                "status": row.get("status", ""),
                "reviewer": row.get("reviewer", ""),
                "notes": row.get("notes", ""),
                "reviewed_at": row.get("reviewed_at", ""),
            }
        )

    approved_total = int(copied_by_split["train"] + copied_by_split["val"])
    if approved_total < int(args.min_approved):
        raise RuntimeError(
            f"Approved samples {approved_total} < required min_approved {int(args.min_approved)}."
        )

    ds_yaml = _write_dataset_yaml(out_root)
    report = {
        "review_csv": review_csv.as_posix(),
        "source_root": source_root.as_posix(),
        "approved_total": approved_total,
        "approved_by_split": copied_by_split,
        "decision_counts": dict(counts),
        "allow_pending_as_approved": bool(args.allow_pending_as_approved),
        "dataset_yaml": ds_yaml.as_posix(),
        "entries": accepted_entries,
    }

    report_path = out_root / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

