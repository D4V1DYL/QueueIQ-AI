"""
fetch_demo_video.py

Download 1-3 supermarket checkout-queue videos from YouTube (via yt-dlp,
no ffmpeg needed — a single mp4 stream <= 480p is chosen) into videos/ so
server.py can play them as a VIRTUAL CAMERA for the /live demo without a
webcam and without a real supermarket.

    python fetch_demo_video.py                 # 2 videos, default queries
    python fetch_demo_video.py --max 3 --query "grocery checkout line overhead"

Note: this is third-party footage; keep it for internal demos.
"""

import argparse
import re
import sys
from pathlib import Path

import yt_dlp

VIDEOS_DIR = Path(__file__).resolve().parent / "videos"
QUERIES = [
    "supermarket checkout queue cctv overhead",
    "grocery store checkout line customers with shopping carts footage",
    "supermarket cashier queue people with baskets b-roll",
]
MIN_SEC, MAX_SEC = 10, 6 * 60
MAX_HEIGHT = 480
# the title must mention a store/checkout context so off-topic search hits are skipped
TITLE_RE = re.compile(r"supermarket|grocery|checkout|cashier|store|queue|retail|shopping", re.I)


def search(query: str, n: int) -> list[dict]:
    opts = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist"}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{n}:{query}", download=False)
    return [e for e in (info.get("entries") or []) if e and e.get("id")]


def download(video_id: str, title: str) -> Path | None:
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in title)[:50].strip().replace(" ", "_")
    out = VIDEOS_DIR / f"{safe}_{video_id}.mp4"
    if out.exists():
        return out
    opts = {
        # video-only mp4 <= 480p first (no ffmpeg needed for merging), then
        # fall back to progressive 360p (format 18)
        "format": f"bv*[ext=mp4][height<={MAX_HEIGHT}]/18/b[ext=mp4][height<={MAX_HEIGHT}]",
        "outtmpl": str(out),
        "quiet": True, "no_warnings": True, "noplaylist": True,
        "max_filesize": 120 * 1024 * 1024, "socket_timeout": 25,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
    except Exception as e:
        print(f"    failed: {e}")
        return None
    return out if out.exists() else None


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=2, help="number of videos (default 2)")
    ap.add_argument("--query", action="append", help="extra search query (repeatable)")
    args = ap.parse_args()
    VIDEOS_DIR.mkdir(exist_ok=True)

    got = 0
    seen = set()
    for q in (args.query or []) + QUERIES:
        if got >= args.max:
            break
        print(f"query: {q!r}")
        try:
            entries = search(q, 8)
        except Exception as e:
            print(f"  search failed: {e}")
            continue
        for e in entries:
            if got >= args.max:
                break
            dur = e.get("duration") or 0
            if e["id"] in seen or not (MIN_SEC <= dur <= MAX_SEC):
                continue
            if not TITLE_RE.search(e.get("title") or ""):
                continue
            seen.add(e["id"])
            print(f"  downloading: {e.get('title', '')[:60]!r} ({dur}s)")
            p = download(e["id"], e.get("title") or e["id"])
            if p:
                got += 1
                print(f"    -> {p.name} ({p.stat().st_size // 1024} KB)")
    print(f"\n{got} video(s) in {VIDEOS_DIR}")


if __name__ == "__main__":
    main()
