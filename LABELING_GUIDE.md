# CSV Labeling Guide — Queue & Basket

## Columns

| Column | Type | Valid values | Notes |
|---|---|---|---|
| `image_id` | text | free, unique per image/frame | Identifier of the source image (file name / video frame) |
| `person_id` | text | P1, P2, P3, ... | Person ID within one image, ordered by position |
| `queue_position` | int | 1 = front of the queue (next to the till) | Queue order, filled in manually while labeling |
| `has_basket` | text | `yes` / `no` | Whether the person visibly carries a basket/trolley |
| `basket_type` | text | `basket` / `cart` / `none` | Type of shopping container |
| `fullness_label` | text | `empty` / `light` / `medium` / `full` / `no_basket_with_items` | Basket fullness class (used to train the classifier) |
| `estimated_item_count` | int | manual estimate of the item count | Used as rough ground truth while there is no real data |
| `predicted_checkout_time_sec` | int | formula / random baseline | **Replaced by the real model after fine-tuning** |
| `actual_checkout_time_sec` | int | empty at first | Filled in later from real data (stopwatch / till log) |
| `prediction_source` | text | `random_baseline` / `heuristic` / `model_v1` etc. | Marks the prediction method version; important for tracking evaluation |
| `notes` | text | free | Extra remarks (prominent items, odd cases, etc.) |

## Random-but-plausible formula (used now, temporarily)

Because there is no real data yet, `predicted_checkout_time_sec` is computed
from `estimated_item_count` plus random noise so it does not look rigidly linear:

```
base_time = 20 + (item_count * 7)       # base seconds: 20 s overhead + 7 s/item
noise     = random(-10%, +10%) of base_time
predicted_checkout_time_sec = base_time + noise
```

Examples:
- 1 item  → base 27 s  → result ~24–30 s
- 4 items → base 48 s  → result ~43–53 s
- 9 items → base 83 s  → result ~75–91 s
- 17 items → base 139 s → result ~125–153 s

This is deliberately **monotonically increasing with the item count** (not pure
random) so the logic stays plausible for a demo even though the numbers are not
from real data yet.

## Fine-tuning path going forward

1. **Now (hackathon):** fill `predicted_checkout_time_sec` with the formula above,
   `prediction_source = random_baseline`, leave `actual_checkout_time_sec` empty.
2. **During a real trial/demo:** every time a customer really finishes checkout,
   record the actual time in `actual_checkout_time_sec`.
3. **Once enough rows have actual times** (ideally 30+), train a simple regression
   (linear regression or a lookup table per item-count range) from
   `estimated_item_count` → `actual_checkout_time_sec`.
4. Change `prediction_source` to `model_v1`, `model_v2`, etc. every time the model
   is retrained so the progress is visible on the dashboard/demo (good for judges —
   it shows the system "learning").
