from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CLASS_COLORS = {
    0: (30, 30, 230),   # root
    1: (240, 80, 40),   # stem
    2: (40, 200, 60),   # leaves
}


def _parse_label_polygons(label_path: Path, width: int, height: int) -> list[tuple[int, np.ndarray]]:
    out: list[tuple[int, np.ndarray]] = []
    if not label_path.exists():
        return out
    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return out
    for row in text.splitlines():
        parts = row.strip().split()
        if len(parts) < 7:
            continue
        try:
            cls_id = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if len(vals) < 6 or (len(vals) % 2) != 0:
            continue
        pts = np.asarray(vals, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0) * float(width)
        pts[:, 1] = np.clip(pts[:, 1], 0.0, 1.0) * float(height)
        pts_i = pts.astype(np.int32)
        if pts_i.shape[0] >= 3:
            out.append((cls_id, pts_i))
    return out


def _draw_overlay(image_path: Path, label_path: Path, out_path: Path, alpha: float) -> bool:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return False
    h, w = image.shape[:2]
    polys = _parse_label_polygons(label_path, width=w, height=h)
    if not polys:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(out_path), image))

    fill = image.copy()
    for cls_id, pts in polys:
        color = CLASS_COLORS.get(int(cls_id), (180, 180, 180))
        cv2.fillPoly(fill, [pts], color=color)
    blended = cv2.addWeighted(fill, float(alpha), image, 1.0 - float(alpha), 0.0)

    for cls_id, pts in polys:
        color = CLASS_COLORS.get(int(cls_id), (180, 180, 180))
        cv2.polylines(blended, [pts], isClosed=True, color=color, thickness=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), blended))


def _load_entries(manifest_path: Path, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = list(payload.get("entries") or [])
    if limit > 0:
        entries = entries[:limit]
    return entries


def _write_review_guide(path: Path, review_csv: Path) -> None:
    text = textwrap.dedent(
        f"""
        # Manual Review Guide (RU)

        Файл для проверки: `{review_csv.as_posix()}`

        1. Откройте CSV в Excel/Google Sheets.
        2. Просмотрите `overlay_path` и исходник `image_path`.
        3. Заполните поле `status`:
           - `approved` — образец корректный для golden-набора
           - `rejected` — образец плохой/шумный/неподходящий
           - `pending` — еще не проверен
        4. При необходимости укажите `reviewer` и `notes`.
        5. После ревью запустите:
           `python -m training.apply_golden_manual_decisions --review-csv {review_csv.as_posix()} --source-root data/golden_rootstem_400 --out data/golden_rootstem_400_validated`

        Важно:
        - Для финального обучения используйте только `approved`.
        - Рекомендуем минимум 300 approved кейсов.
        """
    ).strip() + "\n"
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a manual review pack for golden root/stem candidates.")
    parser.add_argument("--golden-root", type=str, default="data/golden_rootstem_400")
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--out", type=str, default="data/golden_rootstem_400_review")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overlay-alpha", type=float, default=0.35)
    parser.add_argument("--skip-overlays", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    golden_root = Path(args.golden_root)
    manifest_path = Path(args.manifest) if args.manifest else (golden_root / "golden_manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    review_csv = out_root / "manual_review.csv"
    guide_path = out_root / "README_REVIEW_RU.md"
    overlays_root = out_root / "overlays"

    entries = _load_entries(manifest_path=manifest_path, limit=int(args.limit))
    rows: list[dict[str, Any]] = []

    for idx, entry in enumerate(entries):
        split = str(entry.get("split", "train"))
        name = str(entry.get("golden_name", f"sample_{idx:06d}"))
        image_path = Path(str(entry.get("image", "")))
        label_path = Path(str(entry.get("label", "")))
        if not image_path.exists() or not label_path.exists():
            continue

        overlay_path = overlays_root / split / f"{name}_overlay{image_path.suffix.lower()}"
        overlay_ok = False
        if not args.skip_overlays:
            overlay_ok = _draw_overlay(
                image_path=image_path,
                label_path=label_path,
                out_path=overlay_path,
                alpha=float(args.overlay_alpha),
            )

        rows.append(
            {
                "id": idx + 1,
                "split": split,
                "golden_name": name,
                "image_path": image_path.as_posix(),
                "label_path": label_path.as_posix(),
                "overlay_path": overlay_path.as_posix() if overlay_ok else "",
                "priority_score": f"{float(entry.get('score', 0.0)):.6f}",
                "category": str(entry.get("category", "")),
                "reasons": ";".join([str(x) for x in (entry.get("reasons") or [])]),
                "status": "pending",
                "reviewer": "",
                "notes": "",
                "reviewed_at": "",
            }
        )

    fieldnames = [
        "id",
        "split",
        "golden_name",
        "image_path",
        "label_path",
        "overlay_path",
        "priority_score",
        "category",
        "reasons",
        "status",
        "reviewer",
        "notes",
        "reviewed_at",
    ]
    with review_csv.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    _write_review_guide(path=guide_path, review_csv=review_csv)

    summary = {
        "manifest": manifest_path.as_posix(),
        "review_csv": review_csv.as_posix(),
        "review_guide": guide_path.as_posix(),
        "rows": len(rows),
        "overlays_dir": overlays_root.as_posix(),
        "skip_overlays": bool(args.skip_overlays),
    }
    summary_path = out_root / "review_pack_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

