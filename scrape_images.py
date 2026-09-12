"""
scrape_images.py

Image scraper (DuckDuckGo image search, no API key) to SUPPLEMENT the basket
fullness dataset — not a replacement for your own photos.

Results go to dataset/raw/<class>/ with the prefix "scraped_" so they are easy
to tell apart from (and bulk-delete separately from) real photos:
    rm dataset/raw/*/scraped_*.jpg      # delete every scraped image

USAGE:
    python scrape_images.py                    # all classes, 40 images/class
    python scrape_images.py --kelas full       # one class only
    python scrape_images.py --per_kelas 60

AFTER SCRAPING — MANUAL CURATION IS MANDATORY:
    Open every folder and delete images that are the wrong class, shot from
    a very different angle (side-view product/marketing shots), carry a large
    watermark, or are not shopping baskets at all. Search engines miss often.

Automatic filters already applied:
    - not a valid image file -> dropped
    - shortest side < 200 px -> dropped
    - duplicates (identical content hash) -> dropped, across classes too
"""

import argparse
import hashlib
import io
import time
from pathlib import Path

import requests
from PIL import Image
from ddgs import DDGS

RAW_DIR = Path("dataset/raw")
MIN_SIDE = 200          # px, drop tiny images/thumbnails
TIMEOUT = 10            # seconds per download
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) dataset-collector"}

# Queries per class. English queries return far more results.
QUERIES = {
    "empty": [
        "empty shopping basket supermarket",
        "empty grocery basket top view",
        "empty shopping trolley inside view",
    ],
    "light": [
        "shopping basket with few items inside",
        "grocery basket few products top view",
        "shopping cart with a few groceries",
    ],
    "medium": [
        "shopping basket half full groceries",
        "grocery basket with items inside supermarket",
        "half full shopping cart groceries",
    ],
    "full": [
        # avoid the bare phrase "basket full" -> image search drifts to "cat in basket"
        "grocery shopping basket filled with food products",
        "supermarket basket loaded with groceries top view",
        "overflowing shopping cart groceries supermarket",
    ],
    "no_basket_with_items": [
        "person holding groceries in hands supermarket",
        "customer carrying items without basket store",
    ],
}


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def existing_hashes() -> set[str]:
    """Hash every image ALREADY in raw/ (real photos + earlier scrapes) so
    duplicates are not saved again."""
    hashes = set()
    for p in RAW_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            hashes.add(content_hash(p.read_bytes()))
    return hashes


def validate_and_convert(data: bytes) -> bytes | None:
    """Return JPEG bytes if the image is valid and large enough, else None."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        return None
    if min(img.size) < MIN_SIDE:
        return None
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def scrape_class(kelas: str, target: int, seen: set[str]) -> int:
    out_dir = RAW_DIR / kelas
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for query in QUERIES[kelas]:
        if saved >= target:
            break
        print(f"  query: \"{query}\"")
        try:
            results = DDGS().images(query, max_results=target)
        except Exception as e:
            print(f"    search failed ({e}), moving to the next query")
            continue

        for r in results:
            if saved >= target:
                break
            url = r.get("image")
            if not url:
                continue
            try:
                resp = requests.get(url, timeout=TIMEOUT, headers=HEADERS)
                resp.raise_for_status()
            except Exception:
                continue

            jpg = validate_and_convert(resp.content)
            if jpg is None:
                continue
            h = content_hash(jpg)
            if h in seen:
                continue
            seen.add(h)

            (out_dir / f"scraped_{h}.jpg").write_bytes(jpg)
            saved += 1

        time.sleep(1)  # be polite to the search engine between queries

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kelas", choices=list(QUERIES), default=None,
                        help="One class only (default: all)")
    parser.add_argument("--per_kelas", type=int, default=40,
                        help="Target images per class (default 40)")
    args = parser.parse_args()

    if not RAW_DIR.is_dir():
        raise SystemExit(f"{RAW_DIR} does not exist. Run from the QueueIQ project root.")

    classes = [args.kelas] if args.kelas else list(QUERIES)
    seen = existing_hashes()
    print(f"({len(seen)} images already in raw/, duplicates will be skipped)\n")

    total = 0
    for kelas in classes:
        print(f"[{kelas}] target {args.per_kelas}:")
        n = scrape_class(kelas, args.per_kelas, seen)
        total += n
        print(f"  -> {n} images saved\n")

    print(f"Total {total} new images (prefix scraped_).")
    print("\nMANDATORY: curate manually — open every folder, delete wrong-class /")
    print("odd-angle / watermarked images. Delete every scraped image at once with:")
    print("  rm dataset/raw/*/scraped_*.jpg")


if __name__ == "__main__":
    main()
