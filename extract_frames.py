"""
extract_frames.py

Cara tercepat mengumpulkan dataset: REKAM VIDEO, bukan foto satu-satu.
Script ini mengekstrak frame dari video jadi file .jpg langsung ke folder
kelas di dataset/raw/.

Hitungan: video 30 detik @ 1.5 fps = ±45 gambar. Empat video (satu per
kelas) = target 150 foto tercapai dalam beberapa menit rekaman.

CARA PAKAI:
    python extract_frames.py --video keranjang_penuh.mp4 --kelas full
    python extract_frames.py --video vid1.mp4 vid2.mp4 --kelas medium --fps 2

Argumen:
    --video   satu atau beberapa file video (mp4/mov/mkv, hasil rekaman HP)
    --kelas   empty / light / medium / full / no_basket_with_items
    --fps     berapa frame diambil per detik video (default 1.5;
              JANGAN terlalu tinggi — frame yang berdekatan hampir identik,
              tidak menambah informasi untuk model)

Tips merekam biar hasil ekstraknya beneran bervariasi (bukan 45 foto kembar):
    - GERAKKAN kamera pelan-pelan: keliling keranjang, ubah ketinggian & sudut.
    - Di tengah rekaman, acak/geser posisi barang di keranjang.
    - Rekam 2-3 video pendek di lokasi/pencahayaan beda, bukan 1 video panjang.

Butuh ffmpeg (sudah terpasang di sistem ini).
"""

import argparse
import subprocess
import sys
from pathlib import Path

VALID_CLASSES = ["empty", "light", "medium", "full", "no_basket_with_items"]
RAW_DIR = Path("dataset/raw")


def extract(video: Path, out_dir: Path, fps: float) -> int:
    prefix = video.stem.replace(" ", "_")
    pattern = out_dir / f"{prefix}_%04d.jpg"

    existing = set(out_dir.glob(f"{prefix}_*.jpg"))
    if existing:
        print(f"  LEWATI {video.name}: sudah pernah diekstrak "
              f"({len(existing)} frame ada). Hapus dulu kalau mau ulang.")
        return 0

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video),
        "-vf", f"fps={fps}",
        "-q:v", "2",          # kualitas jpg tinggi
        str(pattern),
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  GAGAL mengekstrak {video.name} (ffmpeg error).")
        return 0

    n = len(list(out_dir.glob(f"{prefix}_*.jpg")))
    print(f"  {video.name} -> {n} frame")
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", nargs="+", required=True,
                        help="Satu atau beberapa file video")
    parser.add_argument("--kelas", required=True, choices=VALID_CLASSES,
                        help="Kelas fullness untuk SEMUA video di perintah ini")
    parser.add_argument("--fps", type=float, default=1.5,
                        help="Frame per detik yang diambil (default 1.5)")
    args = parser.parse_args()

    out_dir = RAW_DIR / args.kelas
    if not out_dir.is_dir():
        sys.exit(f"Folder {out_dir} tidak ada. Jalankan dari root proyek QueueIQ.")

    total = 0
    print(f"Ekstrak ke {out_dir}/ @ {args.fps} fps:")
    for v in args.video:
        video = Path(v)
        if not video.is_file():
            print(f"  LEWATI {v}: file tidak ditemukan.")
            continue
        total += extract(video, out_dir, args.fps)

    count_now = len([p for p in out_dir.iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
    print(f"\n+{total} frame baru. Total di {out_dir}/: {count_now} gambar.")
    if total:
        print("PENTING: buka foldernya sebentar, hapus frame yang blur/kembar,")
        print("lalu jalankan: python prepare_dataset.py")


if __name__ == "__main__":
    main()
