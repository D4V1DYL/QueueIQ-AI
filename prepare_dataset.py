"""
prepare_dataset.py

Splits the raw photos in dataset/raw/<class>/ into the train/val structure
required by train_fullness_classifier.py:

    dataset/raw/empty/    --->    dataset/train/empty/  +  dataset/val/empty/
    dataset/raw/light/            dataset/train/light/  +  dataset/val/light/
    ... etc.

USAGE:
1. Put every photo into its class folder under dataset/raw/
   (any file name, formats .jpg/.jpeg/.png/.webp).
2. Run:
       python prepare_dataset.py
   Optional:
       python prepare_dataset.py --val_ratio 0.2 --seed 42
3. Continue with training:
       python train_fullness_classifier.py --data_dir ./dataset --epochs 15

Notes:
- The split is shuffled with a fixed seed -> the same result on every run
  (reproducible). Change --seed for a different split.
- train/ and val/ are DELETED and recreated on every run, so it is safe to
  run repeatedly as photos are added. The originals in raw/ are never
  touched (only copied).
- Classes with 0 photos are skipped; classes with very few photos get a
  warning (about 20 photos per class is the recommended minimum).

Uses only the Python standard library — no torch/pandas needed for this step.
"""

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MIN_RECOMMENDED_PER_CLASS = 20
MIN_VAL_IMAGES = 2  # at least this many per class in val, so val_acc means something


def collect_images(class_dir: Path) -> list[Path]:
    return sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="dataset/raw")
    parser.add_argument("--out_dir", type=str, default="dataset")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Share of photos used for validation (default 20%%)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    rng = random.Random(args.seed)

    if not raw_dir.is_dir():
        raise SystemExit(f"Folder {raw_dir} not found. Run from the project root.")

    class_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir())
    if not class_dirs:
        raise SystemExit(f"No class folders in {raw_dir}.")

    # clear the previous split so no stale photos linger
    for split in ("train", "val"):
        split_dir = out_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)

    print(f"{'Class':<22} {'Total':>6} {'Train':>6} {'Val':>5}")
    print("-" * 45)

    total_all = 0
    warnings = []

    for class_dir in class_dirs:
        images = collect_images(class_dir)
        name = class_dir.name

        if not images:
            warnings.append(f"'{name}': 0 photos — class skipped.")
            continue

        rng.shuffle(images)
        n_val = max(MIN_VAL_IMAGES, round(len(images) * args.val_ratio))
        n_val = min(n_val, len(images) - 1)  # leave at least 1 for train
        val_images, train_images = images[:n_val], images[n_val:]

        for split, subset in (("train", train_images), ("val", val_images)):
            if not subset:
                # an empty class folder makes torchvision's ImageFolder fail,
                # so do not create it at all
                warnings.append(
                    f"'{name}': no photos left for {split} — "
                    f"add photos of this class before training."
                )
                continue
            dest = out_dir / split / name
            dest.mkdir(parents=True, exist_ok=True)
            for img in subset:
                shutil.copy2(img, dest / img.name)

        total_all += len(images)
        print(f"{name:<22} {len(images):>6} {len(train_images):>6} {len(val_images):>5}")

        if len(images) < MIN_RECOMMENDED_PER_CLASS:
            warnings.append(
                f"'{name}': only {len(images)} photos — aim for at least "
                f"{MIN_RECOMMENDED_PER_CLASS} so the model does not just memorise."
            )

    print("-" * 45)
    print(f"{'TOTAL':<22} {total_all:>6}")

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  - {w}")

    if total_all > 0:
        print(f"\nDone. Continue with training:")
        print(f"  python train_fullness_classifier.py --data_dir {out_dir} --epochs 15")


if __name__ == "__main__":
    main()
