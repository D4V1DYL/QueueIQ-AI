# Photo Collection Guide — Basket Fullness Dataset (about 150 photos)

Target: **about 150 photos** to train the basket fullness classifier
(`empty` / `light` / `medium` / `full`). A phone camera is enough.

## 1. Target count per class

Split evenly, keeping a few for the special case:

| Class | Definition | Target photos |
|---|---|---|
| `empty` | empty basket | 30 |
| `light` | 1–5 items | 35 |
| `medium` | 6–15 items | 35 |
| `full` | 16+ items | 35 |
| `no_basket_with_items` | items carried by hand, no basket | 15 (optional but strongly recommended) |

If you cannot reach 150, **prioritise balance between classes** over the
total. 25 balanced photos per class beat 60 `empty` + 15 `full`.

## 2. Camera angle and position

- **Prefer top-down / overhead** (camera above, about 45–90° from horizontal) —
  this matches the real camera position above the checkout queue.
- Distance about 1–2.5 m from the basket, with the whole basket in frame.
- Some photos may be oblique — the real camera is not always perfectly vertical either.

## 3. Variation that MUST be present (this is what makes the model robust)

For every class, vary:

- **Basket contents**: different kinds of items (bottles, packaged goods, produce, boxes).
  Do not shoot 35 `medium` photos with exactly the same contents.
- **Lighting**: bright, slightly dim, artificial light vs window light.
- **Background**: floors of different colour/texture, near the till, in an aisle.
- **Basket type**: hand baskets and trolleys, ideally both.
- **Light occlusion**: in some photos a hand or body partly covers the basket —
  that is how real conditions look.

Avoid: badly blurred photos, baskets cut off by more than half, or duplicates
(a 10-shot burst from the same position is effectively one data point).

## 4. File details

- Format: `.jpg` / `.png` / `.webp` — any.
- File names are free (no renaming needed), but no duplicate names inside one folder.
- Standard phone resolution is enough (resized to 224×224 during training).

## 5. Workflow once the photos are collected

1. Put the photos into their class folders:
   ```
   dataset/raw/empty/
   dataset/raw/light/
   dataset/raw/medium/
   dataset/raw/full/
   dataset/raw/no_basket_with_items/   (if available)
   ```
2. Automatic train/val split (80/20):
   ```
   python prepare_dataset.py
   ```
3. Training (on a laptop/Colab with PyTorch):
   ```
   pip install torch torchvision pillow
   python train_fullness_classifier.py --data_dir ./dataset --epochs 15
   ```
4. Test on a new photo:
   ```
   python predict.py --image ./test_photo.jpg
   ```

Photos can be added incrementally — `prepare_dataset.py` is safe to run again
and again; each run discards the old split and re-splits from `raw/`.
The originals in `raw/` are never modified or deleted.

## 6. When to start training

- **< 20 photos/class**: not yet — keep collecting.
- **20–30 photos/class**: a first training run is possible; rough accuracy, fine for an early demo.
- **30–40 photos/class** (about 150 total): the sweet spot for MobileNetV2 transfer learning.
