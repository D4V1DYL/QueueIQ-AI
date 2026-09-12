"""
scrape_videos.py

Scrape VIDEOS from YouTube (search + download via yt-dlp, no API key) and
extract their frames (ffmpeg) to supplement the basket fullness dataset.

Why video beats stock photos: one video = dozens of frames with different
angles/moments, and lots of real supermarket footage (not white-background
studio shots) — much closer to the overhead camera during the demo.

Frames go to dataset/raw/<class>/ with the prefix "vidscrape_" so they are
easy to tell apart / delete:
    rm dataset/raw/*/vidscrape_*.jpg

USAGE:
    python scrape_videos.py --kelas full
    python scrape_videos.py --kelas _unsorted --max_video 3 --fps 0.5

WARNING — curation is MANDATORY and heavier than for photo scraping:
    A single video often contains all kinds of basket states, yet every frame
    gets the same class label. After scraping, open the folder and delete or
    move the frames that do not match the class.
"""

import argparse
import subprocess
from pathlib import Path

import yt_dlp

RAW_DIR = Path("dataset/raw")
VIDEO_CACHE = Path("dataset/video_cache")   # raw videos are kept here
MAX_DURATION_SEC = 6 * 60                   # skip videos longer than 6 minutes
MIN_DURATION_SEC = 8                        # skip very short videos
MAX_HEIGHT = 480                            # 480p is enough (resized to 224 px for training)
SEARCH_N = 10                               # candidates per query

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
    # bonus: overhead checkout-queue footage — a mix of classes.
    # Frames go to dataset/unsorted/ (NOT raw/, so prepare_dataset.py does
    # not treat it as a class) to be sorted by hand.
    "_unsorted": [
        "supermarket checkout queue cctv overhead",
        "grocery store checkout line overhead camera",
    ],
}


def search_youtube(query: str, n: int) -> list[dict]:
    """Search YouTube via yt-dlp, return a list of {id, title, duration, url}."""
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
        # bv* = allow video-only streams (audio is not needed for frames)
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
            print(f"    search failed ({e})")
            continue

        for r in results:
            if downloaded >= max_video:
                break
            if not (MIN_DURATION_SEC <= r["duration"] <= MAX_DURATION_SEC):
                continue

            prefix = f"vidscrape_{r['id']}"
            if list(out_dir.glob(f"{prefix}_*.jpg")):
                continue  # already extracted into this folder

            video_path = VIDEO_CACHE / f"{r['id']}.mp4"
            if not video_path.exists():
                print(f"    downloading: {r['title'][:60]!r} ({r['duration']}s)")
                if not download_video(r["url"], video_path):
                    print("      failed/skipped")
                    continue

            n = extract_frames(video_path, out_dir, prefix, fps)
            if n:
                downloaded += 1
                total_frames += n
                print(f"      -> {n} frames")

    return total_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kelas", choices=list(QUERIES), required=True)
    parser.add_argument("--max_video", type=int, default=3,
                        help="Maximum videos downloaded per class (default 3)")
    parser.add_argument("--fps", type=float, default=0.5,
                        help="Frames per second of video (default 0.5 = 1 frame every 2 s)")
    args = parser.parse_args()

    if not RAW_DIR.is_dir():
        raise SystemExit(f"{RAW_DIR} does not exist. Run from the QueueIQ project root.")

    print(f"[{args.kelas}]")
    n = scrape_class(args.kelas, args.max_video, args.fps)
    dest = "dataset/unsorted" if args.kelas == "_unsorted" else f"dataset/raw/{args.kelas}"
    print(f"\nTotal {n} new frames in {dest}/ (prefix vidscrape_).")
    print("Raw videos are kept in dataset/video_cache/ (delete to free space).")
    print("MANDATORY curation: delete/move frames whose class does not match.")


if __name__ == "__main__":
    main()
