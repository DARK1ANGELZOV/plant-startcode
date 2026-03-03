from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload YOLO-seg dataset pack to Roboflow project.")
    parser.add_argument("--dataset", default="data/roboflow_upload/plant_2f4ay_pack")
    parser.add_argument("--workspace", default="s-workspace-wlhsh")
    parser.add_argument("--project", default="plant-2f4ay")
    parser.add_argument("--project-type", default="instance-segmentation")
    parser.add_argument("--batch-name", default="expert_pack_v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    if not (dataset_dir / "dataset.yaml").exists():
        raise FileNotFoundError("dataset.yaml not found in dataset directory.")

    if args.dry_run:
        print(f"[DRY RUN] dataset={dataset_dir}")
        print(f"[DRY RUN] workspace={args.workspace} project={args.project}")
        print(f"[DRY RUN] project_type={args.project_type} batch_name={args.batch_name}")
        print("[DRY RUN] API key is not required in dry-run mode.")
        return

    api_key = args.api_key or os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is missing. Set env var or pass --api-key."
        )

    try:
        from roboflow import Roboflow
    except Exception as exc:
        raise RuntimeError(
            "roboflow SDK is not installed. Install with: pip install roboflow"
        ) from exc

    rf = Roboflow(api_key=api_key)
    workspace = rf.workspace(args.workspace)
    # Upload full dataset pack (YOLO-seg layout with dataset.yaml).
    result = workspace.upload_dataset(
        str(dataset_dir),
        args.project,
        project_type=args.project_type,
        batch_name=args.batch_name,
    )
    print(result)


if __name__ == "__main__":
    main()
