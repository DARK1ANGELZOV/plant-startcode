from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import UTC, datetime
from pathlib import Path


VALID_APPROVE = {"a", "approve", "approved", "y", "yes", "ok", "1"}
VALID_REJECT = {"r", "reject", "rejected", "n", "no", "0"}
VALID_PENDING = {"p", "pending", "s", "skip"}
VALID_QUIT = {"q", "quit", "exit"}
VALID_UNSURE = {"u", "unsure"}


def _load_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    if "status" not in fieldnames:
        fieldnames.append("status")
    if "reviewer" not in fieldnames:
        fieldnames.append("reviewer")
    if "notes" not in fieldnames:
        fieldnames.append("notes")
    if "reviewed_at" not in fieldnames:
        fieldnames.append("reviewed_at")
    return rows, fieldnames


def _save_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _maybe_open(path_value: str) -> None:
    if not path_value:
        return
    p = Path(path_value)
    if not p.exists():
        return
    try:
        if os.name == "nt":
            os.startfile(str(p))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            os.system(f'open "{p}"')
        else:
            os.system(f'xdg-open "{p}"')
    except Exception:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI manual review for golden candidate CSV.")
    parser.add_argument("--review-csv", type=str, default="data/golden_rootstem_400_review/manual_review.csv")
    parser.add_argument("--reviewer", type=str, default="reviewer_1")
    parser.add_argument("--from-id", type=int, default=1)
    parser.add_argument("--to-id", type=int, default=0)
    parser.add_argument("--only-pending", action="store_true")
    parser.add_argument("--open-files", action="store_true")
    return parser.parse_args()


def _normalize_decision(raw: str) -> str:
    value = raw.strip().lower()
    if value in VALID_APPROVE:
        return "approved"
    if value in VALID_REJECT:
        return "rejected"
    if value in VALID_PENDING:
        return "pending"
    if value in VALID_UNSURE:
        return "unsure"
    if value in VALID_QUIT:
        return "quit"
    return "invalid"


def main() -> None:
    args = parse_args()
    review_csv = Path(args.review_csv)
    if not review_csv.exists():
        raise FileNotFoundError(f"Review CSV not found: {review_csv}")

    rows, fieldnames = _load_rows(review_csv)
    start_id = max(1, int(args.from_id))
    end_id = int(args.to_id) if int(args.to_id) > 0 else 10**9

    total = len(rows)
    print(f"Loaded {total} rows from {review_csv.as_posix()}")
    print("Commands: [a]pprove / [r]eject / [s]kip / [u]nsure / [q]uit")

    reviewed = 0
    for row in rows:
        row_id = int(float(row.get("id", "0") or 0))
        if row_id < start_id or row_id > end_id:
            continue
        if args.only_pending and str(row.get("status", "")).strip().lower() != "pending":
            continue

        print("\n" + "=" * 72)
        print(f"id={row_id} split={row.get('split','')} name={row.get('golden_name','')}")
        print(f"score={row.get('priority_score','')} category={row.get('category','')}")
        print(f"reasons={row.get('reasons','')}")
        print(f"overlay={row.get('overlay_path','')}")
        print(f"image={row.get('image_path','')}")

        if args.open_files:
            _maybe_open(row.get("overlay_path", ""))
            _maybe_open(row.get("image_path", ""))

        decision_raw = input("decision [a/r/s/u/q]: ").strip()
        decision = _normalize_decision(decision_raw)
        if decision == "quit":
            break
        if decision == "invalid":
            print("Invalid decision, skipping.")
            continue

        notes = input("notes (optional): ").strip()
        row["status"] = decision
        row["reviewer"] = str(args.reviewer)
        row["notes"] = notes
        row["reviewed_at"] = datetime.now(UTC).isoformat()
        reviewed += 1

        _save_rows(review_csv, rows, fieldnames)
        print(f"Saved: id={row_id} -> {decision}")

    _save_rows(review_csv, rows, fieldnames)
    print(f"Done. Updated rows: {reviewed}")


if __name__ == "__main__":
    main()

