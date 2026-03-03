from __future__ import annotations

import argparse
import os
from typing import Iterable

from roboflow import Roboflow


def chunks(seq: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete all images from a Roboflow project.")
    parser.add_argument("--workspace", default="s-workspace-wlhsh")
    parser.add_argument("--project", default="plant-2f4ay")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is missing.")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)

    all_ids: list[str] = []
    offset = 0
    limit = 100
    while True:
        rows = project.search(offset=offset, limit=limit, fields=["id"])
        if not rows:
            break
        ids = [str(r["id"]) for r in rows if "id" in r]
        all_ids.extend(ids)
        if len(rows) < limit:
            break
        offset += len(rows)

    print(f"found_images={len(all_ids)}")
    if args.dry_run:
        print("dry_run=true")
        return

    deleted = 0
    for part in chunks(all_ids, max(1, int(args.batch_size))):
        project.delete_images(part)
        deleted += len(part)
        print(f"deleted={deleted}")

    print(f"done deleted_total={deleted}")


if __name__ == "__main__":
    main()
