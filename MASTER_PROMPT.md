# MASTER PROMPT — AI-Powered Smart Checkout Queue System

Use this prompt in full to brief an AI coding assistant (Claude Code, ChatGPT,
etc.), or as the reference document for the hackathon team.

---

## 1. Project concept

Build an **AI-Powered Smart Checkout Queue System** for a supermarket with
several checkout lanes. Each lane has:
- An overhead (top-down) camera watching the queue.
- A physical three-colour light: 🟢 fastest, 🟡 normal, 🔴 slow.

Goal: use computer vision + prediction to decide which lane is predicted to be
fastest, then communicate it through the physical light and a real-time
dashboard. The score must not just count people — two customers with full
baskets can be slower than five customers with a few items each.

## 2. MVP scope (24–48 hour hackathon)

**Priority — do this first:**
1. Person detection in the queue (pretrained model, NO retraining needed).
2. Cart/basket detection (light fine-tune from a public dataset).
3. Basket fullness classifier: `empty` / `light` / `medium` / `full` (NOT per-item detection).
4. Simple queue tracking (ByteTrack/DeepSORT on top of the person detections).
5. Checkout Load / Queue Score per lane, computed from the number of people + average fullness.
6. Checkout-time prediction: online linear regression that learns from feedback
   (predicted_time = intercept + slope × item_count), updated whenever a new
   actual_checkout_time_sec arrives.
7. Customer-facing dashboard (lane status + wait estimate) and manager-facing
   dashboard (per-lane detail + root cause of why a lane is slow).
8. Physical light control (ESP32/Arduino) that changes colour with the real-time score.

**DO NOT do this in the MVP (too heavy for the time available):**
- Precise per-item object detection inside baskets (occlusion is too heavy, needs 300–500+ images).
- Complex deep learning (LSTM etc.) for time prediction — too little data, it will overfit. Use a simple linear/regression model.

## 3. Computer vision pipeline — technical details

| Component | Approach | Dataset used | Custom data needed |
|---|---|---|---|
| Person detection | YOLOv8/v11 pretrained (COCO), filter class `person` | None | 0 images |
| Cart/basket detection | Fine-tune from a public dataset (Smart Cart 4 / RPC checkout dataset on Roboflow Universe) | Yes, as a base | 30–50 images from your own camera angle |
| Basket fullness | Classifier (MobileNet/ResNet, transfer learning), 4 classes | Partly, or record your own | 80–150 images (20–40/class), or a CV heuristic (background subtraction) for the fastest version, 0 images |
| Queue tracking | Algorithmic (ByteTrack/DeepSORT), not a trained model | None | 0 |
| Checkout-time prediction | Online linear regression (SGD update per transaction) | Synthetic (see section 5) | 20–30 data points for initial calibration |

If time is very tight, basket fullness can be replaced by a classic
computer-vision heuristic (count non-background pixel area inside the basket
bounding box) with no model training at all — trade-off: rougher accuracy, but
0 dataset and 0 training time.

### 3.1 Basket fullness classifier — implementation

Training + inference code is available:

- **`train_fullness_classifier.py`** — transfer learning from MobileNetV2
  (ImageNet pretrained), retraining only the final classifier layer (optionally
  the last few feature blocks with `--unfreeze_last`). Suited to small datasets
  (80–150 images), light, runs on a normal CPU (no GPU required) — important for
  a hackathon demo on a laptop.
  Input: folders `dataset/train/{empty,light,medium,full}/` and
  `dataset/val/{empty,light,medium,full}/`. Output: `fullness_classifier.pt`
  + `class_names.txt`.
- **`predict.py`** — inference on new photos (one file or a whole folder),
  printing the predicted class + confidence per class.

**Usage (on a laptop / Google Colab with PyTorch):**
```
pip install torch torchvision pillow

python train_fullness_classifier.py --data_dir ./dataset --epochs 15
python predict.py --image ./basket_example.jpg
```

If the hackathon deadline is too tight to collect 80–150 labelled photos, fall
back to the CV heuristic (background subtraction / filled pixel area inside the
basket box) as a temporary replacement — 0 dataset, 0 training — and switch
back to this classifier once enough photos exist.

## 4. Label & annotation scheme

**Bounding boxes (YOLO format):**
- `person`
- `cart`, `basket`
- `person-with-cart` (optional)

**Fullness classification (per basket crop):**
- `empty`, `light` (1–5 items), `medium` (6–15 items), `full` (16+ items)
- `no_basket_with_items` (customer carrying items by hand with no basket — MUST be anticipated, it happens often in real conditions)

**CSV log per transaction (columns):**
```
image_id, person_id, queue_position, has_basket, basket_type,
fullness_label, estimated_item_count, predicted_checkout_time_sec,
actual_checkout_time_sec, prediction_source, notes
```

## 5. Data strategy — no real data during the hackathon

Because there is no access to real supermarket data, use **synthetic data that
is "statistically plausible"**, not pure random:

- Generate ~100–120 synthetic transactions: base time from the formula
  `20 + 7 × item_count`, plus cashier-speed variation (normal noise ±8–25%)
  and ~12% of transactions with an outlier delay (30–90 seconds, simulating a
  declined card / payment issue).
- Run the online-learning simulation over this data sequentially (as if it
  arrived one transaction at a time) to prove the learning mechanism works: the
  model parameters must converge towards the formula's true values, and
  prediction accuracy must rise over time (target from ~70% to ~85–90%+).
- Show this "prediction accuracy improvement" chart on the manager dashboard as
  evidence that the system "learns from feedback".
- **Transparency to judges:** explain that this data is synthetic to prove the
  mechanism, and that once deployed the same mechanism learns from real checkout
  data automatically without code changes.

## 6. System architecture

```
OVERHEAD CAMERA
  → COMPUTER VISION (Python, OpenCV + YOLO)
    → Person / Cart / Basket Fullness Detection
      → Queue Tracking (ByteTrack)
        → Prediction Engine (online linear regression)
          → Checkout Load / Wait-Time Score
            ├─→ LED CONTROLLER (ESP32/Arduino) — 🟢🟡🔴
            └─→ DASHBOARD (Next.js)
                 ├─ Customer-facing: lane status + wait estimate
                 └─ Manager-facing: per-lane detail, root cause, accuracy trend
```

**Stack:**
- Computer vision: Python, OpenCV, YOLOv8 (Ultralytics)
- Prediction engine: Python (numpy/pandas), online SGD regression
- Backend/realtime: FastAPI + Server-Sent Events (`server.py`); Supabase optional later
- Frontend: Next.js
- Hardware: ESP32/Arduino for the LEDs, webcam/phone as the overhead camera during the demo
- Communication: SSE / WebSocket

## 7. Database schema (Supabase) — starting point

```sql
-- Checkout transactions (one log row per customer who finished checkout)
create table checkout_transactions (
  id bigint generated always as identity primary key,
  lane_id int not null,
  estimated_item_count int,
  fullness_label text check (fullness_label in ('empty','light','medium','full')),
  predicted_checkout_time_sec numeric,
  actual_checkout_time_sec numeric,
  prediction_source text default 'random_baseline',
  created_at timestamptz default now()
);

-- Real-time lane status
create table lane_status (
  lane_id int primary key,
  current_queue_count int,
  avg_fullness_score numeric,
  predicted_wait_sec numeric,
  status text check (status in ('green','yellow','red')),
  updated_at timestamptz default now()
);

-- Model parameters (for online learning, persisted across restarts)
create table model_parameters (
  id bigint generated always as identity primary key,
  intercept numeric,
  slope numeric,
  updated_at timestamptz default now()
);
```

## 8. Demo flow for the judges

1. Show the live camera / footage → real-time person + basket detection on screen.
2. Show the basket fullness classifier working on several baskets with different contents.
3. Show the dashboard with per-lane scores and the physical LED changing colour automatically.
4. Show the prediction-accuracy-improvement chart from the synthetic-data simulation — explain the online-learning mechanism transparently.
5. Show a "before/after" scenario: the light flips from 🟡 to 🔴 when one lane suddenly gets a customer with a full basket.
6. Close with the root-cause explanation on the manager dashboard (e.g. "Lane 3 is slow because of HIGH BASKET LOAD, not just a high headcount").

## 9. Differentiators (for the pitch)

- Not just a people counter — it weighs **basket contents**, not only headcount.
- The system **learns from real feedback** (online learning), not a static score.
- Transparent about the data limits during the hackathon, while the architecture is ready for real data without code changes.
- Computer vision + prediction + physical hardware (LED) + dashboard combined — end to end, not an isolated ML model.

## 10. Supporting files

- `queue_checkout_labeling_template.csv` / `queue_checkout_labeling_blank.csv` — manual annotation templates.
- `generate_synthetic_data.py` — synthetic checkout data generator.
- `online_learning_simulation.py` — online-learning simulation + accuracy-improvement chart.
- `LABELING_GUIDE.md` — column guide & initial prediction formula.
- `train_fullness_classifier.py` — basket fullness classifier training (MobileNetV2 transfer learning).
- `predict.py` — basket fullness classifier inference on new photos.
- `detect_queue.py` — full queue analysis on an image/video (YOLO + classifier + wait estimate).
- `queueiq_engine.py` + `server.py` — the same pipeline as a library and as the API server behind the `/live` dashboard.

**Current implementation status:** see the Status section in `README.md`.
