"""
finetune_basket_detector.py

Fine-tune detector keranjang (MASTER_PROMPT §3: "Cart/basket detection —
fine-tune ringan"), TANPA anotasi manual, lewat dua tahap:

TAHAP 1 — AUTO-LABEL (bootstrap):
    YOLO-World (open-vocabulary, zero-shot) mendeteksi "shopping basket" /
    "shopping cart" di semua foto keranjang di dataset/raw/, lalu hasilnya
    ditulis sebagai label format YOLO ke dataset_detect/.
    Foto tanpa deteksi dilewati (tidak masuk dataset training).

TAHAP 2 — FINE-TUNE:
    YOLOv8n (6MB, cepat) dilatih dari label tahap 1. Hasilnya detector
    keranjang khusus yang jauh lebih ringan dari YOLO-World untuk dipakai
    real-time di detect_queue.py.

CARA PAKAI:
    python finetune_basket_detector.py                 # label + train
    python finetune_basket_detector.py --skip_label    # train saja (label sudah ada)
    python finetune_basket_detector.py --epochs 60

Output: basket_detector.pt (+ metrik val di runs/detect/).

Catatan: kualitas label = kualitas YOLO-World, bukan manusia — cukup untuk
MVP hackathon. Saat ada waktu, review label di dataset_detect/labels/
(format: class cx cy w h ternormalisasi) atau tambah foto sendiri.
"""

import argparse
import random
import shutil
from pathlib import Path

from ultralytics import YOLO

RAW_CLASSES = ["empty", "light", "medium", "full"]   # folder berisi keranjang
PROMPTS = ["shopping basket", "shopping cart"]        # kelas 0 = basket, 1 = cart
DETECT_DIR = Path("dataset_detect")
VAL_RATIO = 0.15
SEED = 42
CONF_LABEL = 0.25    # ambang confidence YOLO-World untuk dijadikan label
OUT_MODEL = "basket_detector.pt"


def autolabel():
    world = YOLO("yolov8s-worldv2.pt")
    world.set_classes(PROMPTS)

    images = []
    for cls in RAW_CLASSES:
        images += sorted((Path("dataset/raw") / cls).glob("*.jpg"))
    rng = random.Random(SEED)
    rng.shuffle(images)

    n_val = max(5, int(len(images) * VAL_RATIO))
    split_of = {p: ("val" if i < n_val else "train") for i, p in enumerate(images)}

    if DETECT_DIR.exists():
        shutil.rmtree(DETECT_DIR)
    for split in ("train", "val"):
        (DETECT_DIR / "images" / split).mkdir(parents=True)
        (DETECT_DIR / "labels" / split).mkdir(parents=True)

    kept = {"train": 0, "val": 0}
    skipped = 0
    for p in images:
        r = world.predict(str(p), conf=CONF_LABEL, verbose=False)[0]
        if not len(r.boxes):
            skipped += 1
            continue
        split = split_of[p]
        # nama unik: <kelasfolder>_<namafile>
        stem = f"{p.parent.name}_{p.stem}"
        shutil.copy2(p, DETECT_DIR / "images" / split / f"{stem}.jpg")
        lines = []
        for b in r.boxes:
            cx, cy, w, h = b.xywhn[0].tolist()
            lines.append(f"{int(b.cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (DETECT_DIR / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
        kept[split] += 1

    (DETECT_DIR / "data.yaml").write_text(
        f"path: {DETECT_DIR.resolve()}\n"
        "train: images/train\nval: images/val\n"
        f"names:\n  0: basket\n  1: cart\n"
    )
    print(f"\nAuto-label selesai: train={kept['train']} val={kept['val']} "
          f"(dilewati tanpa deteksi: {skipped})")


def train(epochs: int, batch: int):
    model = YOLO("yolov8n.pt")   # mulai dari pretrained COCO
    results = model.train(
        data=str(DETECT_DIR / "data.yaml"),
        epochs=epochs, batch=batch, imgsz=640,
        patience=15, seed=SEED, verbose=False,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    shutil.copy2(best, OUT_MODEL)
    print(f"\nModel tersimpan -> {OUT_MODEL}")
    print(f"Metrik lengkap  -> {results.save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_label", action="store_true",
                        help="Lewati auto-label (dataset_detect/ sudah ada)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    if not args.skip_label:
        autolabel()
    train(args.epochs, args.batch)


if __name__ == "__main__":
    main()
