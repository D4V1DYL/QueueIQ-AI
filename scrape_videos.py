"""
scrape_videos.py

Scrape VIDEO dari YouTube (search + unduh via yt-dlp, tanpa API key) lalu
ekstrak frame-nya (ffmpeg) untuk mensuplemen dataset basket fullness.

Kenapa video lebih bagus dari foto stok: satu video = puluhan frame dengan
sudut/momen berbeda, dan banyak footage supermarket beneran (bukan foto
studio latar putih) — lebih mirip kondisi kamera overhead saat demo.

Hasil frame disimpan ke dataset/raw/<kelas>/ dengan prefix "vidscrape_"
supaya gampang dibedakan/dihapus:
    rm dataset/raw/*/vidscrape_*.jpg

CARA PAKAI:
    python scrape_videos.py --kelas full
    python scrape_videos.py --kelas _unsorted --max_video 3 --fps 0.5

PERHATIAN — kurasi WAJIB dan lebih berat dari scrape foto:
    Satu video sering berisi macam-macam kondisi keranjang, padahal semua
    frame-nya dilabeli satu kelas. Setelah scrape, buka foldernya dan
    hapus/pindahkan frame yang tidak sesuai kelasnya.
"""

import argparse
import subprocess
from pathlib import Path

import yt_dlp

RAW_DIR = Path("dataset/raw")
VIDEO_CACHE = Path("dataset/video_cache")   # video mentah disimpan di sini
MAX_DURATION_SEC = 6 * 60                   # lewati video > 6 menit
MIN_DURATION_SEC = 8                        # lewati video kelewat pendek
MAX_HEIGHT = 480                            # 480p cukup (di-resize 224px saat training)
SEARCH_N = 10                               # kandidat per query

QUERIES = {
    "empty": [
        "empty shopping basket supermarket footage",
        "supermarket basket stack b-roll",
    ],
    "light": [
        "putting few groceries in shopping basket",
        "shopping basket few items supermarket b-roll",
    ],
    "medium": [
        "grocery shopping filling basket supermarket b-roll",
        "shopping basket groceries supermarket footage",
    ],
    "full": [
        "full shopping basket groceries b-roll",
        "supermarket checkout unloading full basket",
    ],
    "no_basket_with_items": [
        "person carrying groceries in hands supermarket footage",
    ],
    # bonus: footage antrian kasir dari atas — isinya campuran kelas.
    # Frame masuk ke dataset/unsorted/ (BUKAN raw/, biar tidak dianggap
    # kelas oleh prepare_dataset.py) untuk dipilah manual.
    "_unsorted": [
        "supermarket checkout queue cctv overhead",
        "grocery store checkout line overhead camera",
    ],
}


def search_youtube(query: str, n: int) -> list[dict]:
    """Cari YouTube via yt-dlp, return list {id, title, duration, url}."""
    opts = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist"}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{n}:{query}", download=False)
    out = []
    for e in info.get("entries") or []:
        if not e or not e.get("id"):
            continue
        out.append({
            "id": e["id"],
            "title": e.get("title") or "",
            "duration": e.get("duration") or 0,
            "url": e.get("url") or f"https://www.youtube.com/watch?v={e['id']}",
        })
    return out


def download_video(url: str, out_path: Path) -> bool:
    opts = {
        # bv* = izinkan stream video-only (audio tidak dibutuhkan untuk frame)
        "format": f"bv*[height<={MAX_HEIGHT}]/b[height<={MAX_HEIGHT}]/w",
        "outtmpl": str(out_path),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "max_filesize": 200 * 1024 * 1024,
        "socket_timeout": 20,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception:
        return False
    return out_path.exists()


def extract_frames(video: Path, out_dir: Path, prefix: str, fps: float) -> int:
    pattern = out_dir / f"{prefix}_%04d.jpg"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video), "-vf", f"fps={fps}", "-q:v", "2", str(pattern),
    ]
    if subprocess.run(cmd).returncode != 0:
        return 0
    return len(list(out_dir.glob(f"{prefix}_*.jpg")))


def scrape_class(kelas: str, max_video: int, fps: float) -> int:
    if kelas == "_unsorted":
        out_dir = Path("dataset/unsorted")
    else:
        out_dir = RAW_DIR / kelas
    out_dir.mkdir(parents=True, exist_ok=True)
    VIDEO_CACHE.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    downloaded = 0

    for query in QUERIES[kelas]:
        if downloaded >= max_video:
            break
        print(f"  query: \"{query}\"")
        try:
            results = search_youtube(query, SEARCH_N)
        except Exception as e:
            print(f"    search gagal ({e})")
            continue

        for r in results:
            if downloaded >= max_video:
                break
            if not (MIN_DURATION_SEC <= r["duration"] <= MAX_DURATION_SEC):
                continue

            prefix = f"vidscrape_{r['id']}"
            if list(out_dir.glob(f"{prefix}_*.jpg")):
                continue  # sudah pernah diekstrak ke folder ini

            video_path = VIDEO_CACHE / f"{r['id']}.mp4"
            if not video_path.exists():
                print(f"    unduh: {r['title'][:60]!r} ({r['duration']}s)")
                if not download_video(r["url"], video_path):
                    print("      gagal/lewati")
                    continue

            n = extract_frames(video_path, out_dir, prefix, fps)
            if n:
                downloaded += 1
                total_frames += n
                print(f"      -> {n} frame")

    return total_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kelas", choices=list(QUERIES), required=True)
    parser.add_argument("--max_video", type=int, default=3,
                        help="Maksimal video diunduh per kelas (default 3)")
    parser.add_argument("--fps", type=float, default=0.5,
                        help="Frame per detik video (default 0.5 = 1 frame tiap 2 dtk)")
    args = parser.parse_args()

    if not RAW_DIR.is_dir():
        raise SystemExit(f"{RAW_DIR} tidak ada. Jalankan dari root proyek QueueIQ.")

    print(f"[{args.kelas}]")
    n = scrape_class(args.kelas, args.max_video, args.fps)
    dest = "dataset/unsorted" if args.kelas == "_unsorted" else f"dataset/raw/{args.kelas}"
    print(f"\nTotal {n} frame baru di {dest}/ (prefix vidscrape_).")
    print("Video mentah tersimpan di dataset/video_cache/ (hapus kalau butuh ruang).")
    print("WAJIB kurasi: hapus/pindah frame yang kelasnya tidak sesuai.")


if __name__ == "__main__":
    main()
