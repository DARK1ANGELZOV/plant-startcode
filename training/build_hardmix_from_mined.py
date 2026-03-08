from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_img(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def load_pairs(root: Path, split: str) -> list[tuple[Path, Path]]:
    labels_dir = root / "labels" / split
    images_dir = root / "images" / split
    if not labels_dir.exists() or not images_dir.exists():
        return []
    pairs = []
    for lbl in sorted(labels_dir.glob("*.txt")):
        img = find_img(images_dir, lbl.stem)
        if img is None:
            continue
        pairs.append((img, lbl))
    return pairs


def copy_pairs(
    pairs: list[tuple[Path, Path]],
    out_root: Path,
    split: str,
    prefix: str,
    cap: int | None = None,
    repeat: int = 1,
) -> int:
    if cap is not None and len(pairs) > cap:
        pairs = random.sample(pairs, cap)
    n = 0
    for rep in range(repeat):
        for i, (img, lbl) in enumerate(pairs):
            name = f"{prefix}_r{rep}_{i:06d}"
            out_img = out_root / "images" / split / f"{name}{img.suffix.lower()}"
            out_lbl = out_root / "labels" / split / f"{name}.txt"
            shutil.copy2(img, out_img)
            shutil.copy2(lbl, out_lbl)
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge hard-mined + curated datasets for hardmix fine-tuning.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/hf_multisource_hardmix")
    parser.add_argument("--hard", type=str, default="data/hard_mined_rootstem")
    parser.add_argument("--mega-fast", type=str, default="data/hf_multisource_mega10_fast")
    parser.add_argument("--pack", type=str, default="data/roboflow_upload/plant_2f4ay_pack")
    parser.add_argument("--gold", type=str, default="data/roboflow_upload/plant_2f4ay_gold_pack")
    parser.add_argument("--cap-mega-train", type=int, default=900)
    parser.add_argument("--cap-mega-val", type=int, default=220)
    parser.add_argument("--cap-pack-train", type=int, default=700)
    parser.add_argument("--cap-pack-val", type=int, default=90)
    parser.add_argument("--cap-gold-train", type=int, default=420)
    parser.add_argument("--cap-gold-val", type=int, default=72)
    parser.add_argument("--repeat-hard-train", type=int, default=3)
    args = parser.parse_args()

    random.seed(args.seed)
    out_root = Path(args.out)
    if out_root.exists():
        shutil.rmtree(out_root)
    (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    copied = {}

    hard_root = Path(args.hard)
    hard_train = load_pairs(hard_root, "train")
    hard_val = load_pairs(hard_root, "val")
    copied["hard_train"] = copy_pairs(
        hard_train,
        out_root,
        "train",
        "hard",
        cap=None,
        repeat=max(1, int(args.repeat_hard_train)),
    )
    copied["hard_val"] = copy_pairs(hard_val, out_root, "val", "hard", cap=None, repeat=1)

    mega_root = Path(args.mega_fast)
    copied["mega_train"] = copy_pairs(
        load_pairs(mega_root, "train"),
        out_root,
        "train",
        "mega",
        cap=int(args.cap_mega_train),
        repeat=1,
    )
    copied["mega_val"] = copy_pairs(
        load_pairs(mega_root, "val"),
        out_root,
        "val",
        "mega",
        cap=int(args.cap_mega_val),
        repeat=1,
    )

    pack_root = Path(args.pack)
    copied["pack_train"] = copy_pairs(
        load_pairs(pack_root, "train"),
        out_root,
        "train",
        "pack",
        cap=int(args.cap_pack_train),
        repeat=1,
    )
    copied["pack_val"] = copy_pairs(
        load_pairs(pack_root, "val"),
        out_root,
        "val",
        "pack",
        cap=int(args.cap_pack_val),
        repeat=1,
    )

    gold_root = Path(args.gold)
    copied["gold_train"] = copy_pairs(
        load_pairs(gold_root, "train"),
        out_root,
        "train",
        "gold",
        cap=int(args.cap_gold_train),
        repeat=1,
    )
    copied["gold_val"] = copy_pairs(
        load_pairs(gold_root, "val"),
        out_root,
        "val",
        "gold",
        cap=int(args.cap_gold_val),
        repeat=1,
    )

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

    report = {
        "seed": args.seed,
        "copied": copied,
        "total_train": sum(v for k, v in copied.items() if k.endswith("_train")),
        "total_val": sum(v for k, v in copied.items() if k.endswith("_val")),
        "dataset_yaml": str(ds_yaml),
    }
    report_path = out_root / "build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
