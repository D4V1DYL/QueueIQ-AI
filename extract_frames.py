"""
extract_frames.py

The fastest way to collect a dataset: RECORD VIDEO instead of taking photos
one by one. This script extracts frames from a video as .jpg files straight
into the class folder under dataset/raw/.

Maths: a 30-second video at 1.5 fps = about 45 images. Four videos (one per
class) reach the 150-photo target in a few minutes of recording.

USAGE:
    python extract_frames.py --video full_basket.mp4 --kelas full
    python extract_frames.py --video vid1.mp4 vid2.mp4 --kelas medium --fps 2

Arguments:
    --video   one or more video files (mp4/mov/mkv, recorded on a phone)
    --kelas   empty / light / medium / full / no_basket_with_items
    --fps     frames taken per second of video (default 1.5;
              do NOT go too high — neighbouring frames are near-identical
              and add no information for the model)

Recording tips so the extracted frames really vary (not 45 twins):
    - MOVE the camera slowly: walk around the basket, change height and angle.
    - Halfway through, shuffle or move the items in the basket.
    - Record 2-3 short videos in different places/lighting, not one long video.

Requires ffmpeg on the PATH.
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
        print(f"  SKIP {video.name}: already extracted "
              f"({len(existing)} frames present). Delete them first to redo.")
        return 0

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video),
        "-vf", f"fps={fps}",
        "-q:v", "2",          # high JPEG quality
        str(pattern),
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  FAILED to extract {video.name} (ffmpeg error).")
        return 0

    n = len(list(out_dir.glob(f"{prefix}_*.jpg")))
    print(f"  {video.name} -> {n} frames")
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", nargs="+", required=True,
                        help="One or more video files")
    parser.add_argument("--kelas", required=True, choices=VALID_CLASSES,
                        help="Fullness class for ALL videos in this command")
    parser.add_argument("--fps", type=float, default=1.5,
                        help="Frames per second to take (default 1.5)")
    args = parser.parse_args()

    out_dir = RAW_DIR / args.kelas
    if not out_dir.is_dir():
        sys.exit(f"Folder {out_dir} does not exist. Run from the QueueIQ project root.")

    total = 0
    print(f"Extracting into {out_dir}/ @ {args.fps} fps:")
    for v in args.video:
        video = Path(v)
        if not video.is_file():
            print(f"  SKIP {v}: file not found.")
            continue
        total += extract(video, out_dir, args.fps)

    count_now = len([p for p in out_dir.iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
    print(f"\n+{total} new frames. Total in {out_dir}/: {count_now} images.")
    if total:
        print("IMPORTANT: open the folder briefly, delete blurred/duplicate frames,")
        print("then run: python prepare_dataset.py")


if __name__ == "__main__":
    main()
