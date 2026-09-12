"""
finetune_basket_detector.py

Fine-tune a basket detector (MASTER_PROMPT §3: "Cart/basket detection —
light fine-tune") WITHOUT manual annotation, in two stages:

STAGE 1 — AUTO-LABEL (bootstrap):
    YOLO-World (open-vocabulary, zero-shot) detects "shopping basket" /
    "shopping cart" in every basket photo under dataset/raw/, and the results
    are written as YOLO-format labels into dataset_detect/.
    Photos with no detection are skipped (not added to the training set).

STAGE 2 — FINE-TUNE:
    YOLOv8n (6 MB, fast) is trained on the stage-1 labels. The result is a
    dedicated basket detector, far lighter than YOLO-World, for real-time use
    in detect_queue.py and server.py.

USAGE:
    python finetune_basket_detector.py                 # label + train
    python finetune_basket_detector.py --skip_label    # train only (labels already exist)
    python finetune_basket_detector.py --epochs 60

Output: basket_detector.pt (+ validation metrics under runs/detect/).

Note: label quality = YOLO-World quality, not a human's — good enough for a
hackathon MVP. When there is time, review the labels in dataset_detect/labels/
(format: class cx cy w h, normalised) or add your own photos.
"""

import argparse
import random
import shutil
from pathlib import Path

from ultralytics import YOLO

RAW_CLASSES = ["empty", "light", "medium", "full"]   # folders that contain baskets
PROMPTS = ["shopping basket", "shopping cart"]        # class 0 = basket, 1 = cart
DETECT_DIR = Path("dataset_detect")
VAL_RATIO = 0.15
SEED = 42
CONF_LABEL = 0.25    # YOLO-World confidence threshold to accept a label
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
        # unique name: <classfolder>_<filename>
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
    print(f"\nAuto-labelling done: train={kept['train']} val={kept['val']} "
          f"(skipped without detection: {skipped})")


def train(epochs: int, batch: int):
    model = YOLO("yolov8n.pt")   # start from COCO pretrained
    results = model.train(
        data=str(DETECT_DIR / "data.yaml"),
        epochs=epochs, batch=batch, imgsz=640,
        patience=15, seed=SEED, verbose=False,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    shutil.copy2(best, OUT_MODEL)
    print(f"\nModel saved -> {OUT_MODEL}")
    print(f"Full metrics -> {results.save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_label", action="store_true",
                        help="Skip auto-labelling (dataset_detect/ already exists)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    if not args.skip_label:
        autolabel()
    train(args.epochs, args.batch)


if __name__ == "__main__":
    main()
