"""
scrape_images.py

Scraper gambar dari image search (DuckDuckGo, tanpa API key) untuk
MENSUPLEMEN dataset basket fullness — bukan pengganti foto sendiri.

Hasil disimpan ke dataset/raw/<kelas>/ dengan prefix "scraped_" supaya
gampang dibedakan (dan dihapus massal) dari foto asli:
    rm dataset/raw/*/scraped_*.jpg      # hapus semua hasil scrape

CARA PAKAI:
    python scrape_images.py                    # semua kelas, 40 gambar/kelas
    python scrape_images.py --kelas full       # satu kelas saja
    python scrape_images.py --per_kelas 60

SETELAH SCRAPE — WAJIB KURASI MANUAL:
    Buka tiap folder, hapus gambar yang: salah kelas, sudut terlalu beda
    (foto produk/marketing dari samping), ada watermark besar, atau bukan
    keranjang belanja sama sekali. Search engine sering meleset.

Filter otomatis yang sudah diterapkan:
    - file bukan gambar valid -> buang
    - resolusi < 200px sisi terpendek -> buang
    - duplikat (hash konten sama) -> buang, termasuk antar kelas
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
MIN_SIDE = 200          # px, buang gambar kekecilan/thumbnail
TIMEOUT = 10            # detik per download
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) dataset-collector"}

# Query per kelas. Bahasa Inggris -> hasil jauh lebih banyak.
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
        # hindari frasa "basket full" polos -> image search nyasar ke "cat in basket"
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
    """Hash semua gambar yang SUDAH ada di raw/ (foto asli + scrape lama),
    supaya tidak menyimpan duplikat."""
    hashes = set()
    for p in RAW_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            hashes.add(content_hash(p.read_bytes()))
    return hashes


def validate_and_convert(data: bytes) -> bytes | None:
    """Return bytes JPG kalau gambar valid & cukup besar, else None."""
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
            print(f"    search gagal ({e}), lanjut query berikutnya")
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

        time.sleep(1)  # sopan ke search engine antar query

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kelas", choices=list(QUERIES), default=None,
                        help="Satu kelas saja (default: semua)")
    parser.add_argument("--per_kelas", type=int, default=40,
                        help="Target gambar per kelas (default 40)")
    args = parser.parse_args()

    if not RAW_DIR.is_dir():
        raise SystemExit(f"{RAW_DIR} tidak ada. Jalankan dari root proyek QueueIQ.")

    classes = [args.kelas] if args.kelas else list(QUERIES)
    seen = existing_hashes()
    print(f"({len(seen)} gambar sudah ada di raw/, duplikat akan dilewati)\n")

    total = 0
    for kelas in classes:
        print(f"[{kelas}] target {args.per_kelas}:")
        n = scrape_class(kelas, args.per_kelas, seen)
        total += n
        print(f"  -> {n} gambar tersimpan\n")

    print(f"Total {total} gambar baru (prefix scraped_).")
    print("\nWAJIB: kurasi manual — buka tiap folder, hapus yang salah kelas /")
    print("sudut aneh / watermark. Hapus semua hasil scrape sekaligus dengan:")
    print("  rm dataset/raw/*/scraped_*.jpg")


if __name__ == "__main__":
    main()
