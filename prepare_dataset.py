"""
prepare_dataset.py

Membagi foto mentah di dataset/raw/<kelas>/ menjadi struktur train/val
yang dibutuhkan train_fullness_classifier.py:

    dataset/raw/empty/    --->    dataset/train/empty/  +  dataset/val/empty/
    dataset/raw/light/            dataset/train/light/  +  dataset/val/light/
    ... dst

CARA PAKAI:
1. Taruh semua foto ke folder kelasnya masing-masing di dataset/raw/
   (nama file bebas, format .jpg/.jpeg/.png/.webp).
2. Jalankan:
       python prepare_dataset.py
   Opsional:
       python prepare_dataset.py --val_ratio 0.2 --seed 42
3. Lanjut training:
       python train_fullness_classifier.py --data_dir ./dataset --epochs 15

Catatan:
- Split di-shuffle dengan seed tetap -> hasil sama tiap dijalankan ulang
  (reproducible). Ganti --seed kalau mau split berbeda.
- Folder train/ dan val/ DIHAPUS dan dibuat ulang tiap run, jadi aman
  dijalankan berulang kali saat foto bertambah. Foto asli di raw/ tidak
  pernah disentuh (hanya di-copy).
- Kelas dengan 0 foto dilewati; kelas dengan foto sangat sedikit diberi
  peringatan (minimal disarankan ~20 foto per kelas).

Hanya pakai library standar Python — tidak butuh torch/pandas untuk step ini.
"""

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MIN_RECOMMENDED_PER_CLASS = 20
MIN_VAL_IMAGES = 2  # tiap kelas minimal segini di val, biar val_acc bermakna


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
                        help="Porsi foto untuk validasi (default 20%%)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    rng = random.Random(args.seed)

    if not raw_dir.is_dir():
        raise SystemExit(f"Folder {raw_dir} tidak ditemukan. Jalankan dari root proyek.")

    class_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir())
    if not class_dirs:
        raise SystemExit(f"Tidak ada folder kelas di {raw_dir}.")

    # bersihkan hasil split lama supaya tidak ada foto "nyangkut"
    for split in ("train", "val"):
        split_dir = out_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)

    print(f"{'Kelas':<22} {'Total':>6} {'Train':>6} {'Val':>5}")
    print("-" * 45)

    total_all = 0
    warnings = []

    for class_dir in class_dirs:
        images = collect_images(class_dir)
        name = class_dir.name

        if not images:
            warnings.append(f"'{name}': 0 foto — kelas dilewati.")
            continue

        rng.shuffle(images)
        n_val = max(MIN_VAL_IMAGES, round(len(images) * args.val_ratio))
        n_val = min(n_val, len(images) - 1)  # sisakan minimal 1 untuk train
        val_images, train_images = images[:n_val], images[n_val:]

        for split, subset in (("train", train_images), ("val", val_images)):
            if not subset:
                # folder kelas kosong bikin ImageFolder (torchvision) error,
                # jadi jangan dibuat sama sekali
                warnings.append(
                    f"'{name}': tidak kebagian foto untuk {split} — "
                    f"tambah foto kelas ini sebelum training."
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
                f"'{name}': cuma {len(images)} foto — usahakan minimal "
                f"{MIN_RECOMMENDED_PER_CLASS} biar model tidak asal hafal."
            )

    print("-" * 45)
    print(f"{'TOTAL':<22} {total_all:>6}")

    if warnings:
        print("\nPERINGATAN:")
        for w in warnings:
            print(f"  - {w}")

    if total_all > 0:
        print(f"\nSelesai. Lanjut training:")
        print(f"  python train_fullness_classifier.py --data_dir {out_dir} --epochs 15")


if __name__ == "__main__":
    main()
