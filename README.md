# QueueIQ — AI-Powered Smart Checkout Queue System

Smart checkout queue system: an overhead camera plus computer vision rates
**what is in the basket** (not just how many people are in line) to predict
the fastest lane, then communicates it through 🟢🟡🔴 lights and a real-time
dashboard.

**Start here:** [`QueueIQ_Pipeline.ipynb`](QueueIQ_Pipeline.ipynb) — the
end-to-end notebook (dataset → training → evaluation → checkout-time
prediction) with an explanation of every stage. Full concept and
architecture: [`MASTER_PROMPT.md`](MASTER_PROMPT.md).

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

> **cu126 is required** for Pascal GPUs (GTX 10xx) — PyTorch builds for
> cu128+ dropped Pascal support. A GPU is optional: inference runs at ~32 fps on CPU.

## Workflow

| Step | Command |
|---|---|
| Collect photos (video → frames) | `python extract_frames.py --video vid.mp4 --kelas full` |
| Supplement from the internet (optional) | `python scrape_images.py` / `python scrape_videos.py --kelas _unsorted` |
| Split train/val | `python prepare_dataset.py` |
| Training (about 1 minute on GPU) | `python train_fullness_classifier.py --data_dir ./dataset --epochs 15` |
| Predict | `python predict.py --image photo.jpg` |
| **Full queue analysis** (YOLO + classifier + wait estimate) | `python detect_queue.py --image examples/antrian_cctv.jpg` |
| — with a queue zone (count only people inside the polygon) | `python detect_queue.py --image cctv.jpg --zone "130,40 330,40 330,330 130,330"` |
| Online-learning simulation | `python online_learning_simulation.py` |
| **API server for the dashboard** (`QueueIQ-FE` page `/live`) | `pip install -r requirements-server.txt` then `python server.py` |
| CCTV footage for the dashboard's virtual camera | `pip install yt-dlp` then `python fetch_demo_video.py` |

## API server (the bridge to the frontend)

`server.py` wraps the pipeline (`queueiq_engine.py`) as HTTP + Server-Sent
Events on `http://127.0.0.1:8000`, consumed by the **/live** page in QueueIQ-FE.

```bash
python server.py                  # picks the tier automatically from the models present
python server.py --host 0.0.0.0   # demo from a phone / another laptop on the same Wi-Fi
python server.py --mode mock      # no torch: UI demo only
```

| Tier (reported by `/health` and the dashboard header) | Condition |
|---|---|
| `full` | `yolov8n.pt` + `fullness_classifier.pt` + `class_names.txt` present (+ `basket_detector.pt` when available) — **all of them ship with this repo**, just clone |
| `heuristic` | YOLO present, no classifier — fullness from an edge/colour heuristic (MASTER_PROMPT §3) |
| `mock` | torch/ultralytics not installed — deterministic placeholder detections |

Endpoints: `GET /health` · `GET /api/lanes` · `POST /api/lanes/{id}/analyze`
(multipart `file` or form `example=antrian_cctv.jpg`) · `GET /api/lanes/{id}/frame.jpg`
· `POST /api/lanes/{id}/complete` `{"actual_sec": 87}` (feedback → SGD step, persisted in
`model_state.json`) · `POST /api/lanes/{id}/open` · `GET /api/model` (parameters + accuracy
history) · `POST /api/model/reset` · `POST /api/demo/scenario` `{"name": "seed"|"surge"|"cctv"|"clear"}`
· `GET /api/led/{id}` (plain text `green|amber|red|closed` for an ESP32) · `GET /api/events` (SSE).

**Virtual camera (demo without a webcam or a supermarket):** `python fetch_demo_video.py`
downloads checkout CCTV footage from YouTube into `videos/` (2 videos already ship with
the repo). On the `/live` dashboard they appear under "Sample footage · play as virtual
camera"; the server reads one frame every 2 seconds through the same pipeline as a webcam
(`POST /api/lanes/{id}/video`). Drop other `.mp4` files into `videos/` to list them too.

**Feedback guard:** the *Done* button uses the time measured since the front shopper
started being served; under 15 seconds (an operator clicking during a demo) the
transaction is not treated as a real checkout and the model is NOT updated — use
*Teach the model* with an explicit number of seconds instead.

Sample photos for the dashboard gallery come from `examples/` (every `.jpg/.png/.webp`
except `*_annotated`) — put your own photos of queues with carts or baskets there and they
show up as clickable examples.

The server only serves loopback / private-LAN clients; narrow it further with
`QUEUEIQ_ALLOWED_IPS=192.168.1.20,192.168.1.55`. Light thresholds match the
dashboard: green ≤ 120 s, amber ≤ 240 s, red above that.

Training photos go into `dataset/raw/{empty,light,medium,full,no_basket_with_items}/`
(folder name = label). Photo guidelines: [`PHOTO_GUIDE.md`](PHOTO_GUIDE.md); CSV labeling: [`LABELING_GUIDE.md`](LABELING_GUIDE.md).

## Status

- ✅ Basket fullness classifier (MobileNetV2 transfer learning, 9.2 MB, 5.6 ms/image on a GTX 1050 Ti). Retrained from scratch on scraped data (`scrape_images.py` → curation → `prepare_dataset.py` → `train_fullness_classifier.py --epochs 25 --lr 4e-4 --unfreeze_last 4`): 225 photos, 5-class val accuracy 56%, empty-vs-has-items 93%, 3-level (empty / light / medium-full) 80%, within one level 98% — stock-photo labels are noisy; real photos from your own camera will do much better
- ✅ Checkout-time prediction — online linear regression, accuracy 74% → 88% over 120 simulated transactions
- ✅ Person detection + queue score (`detect_queue.py`): pretrained YOLOv8n → carry-region crop per person → fullness → wait estimate → lane status 🟢🟡🔴
- ✅ Fine-tuned basket detector (`finetune_basket_detector.py`): YOLO-World auto-labelling (zero-shot, no manual annotation) → YOLOv8n fine-tune — mAP50 0.887, precision 0.93 on the original dataset; retrained on scraped data (116 photos, 30 epochs on CPU in about 12 minutes) gives mAP50 0.82, precision 0.82; `detect_queue.py` and `server.py` use it automatically when `basket_detector.pt` exists
- ✅ Queue zone (`--zone`): only people/baskets inside the queue polygon are counted — cashiers and passers-by are filtered out
- ✅ API server (`server.py` + `queueiq_engine.py`): FastAPI + SSE, checkout feedback → live online learning, LED endpoint for an ESP32, connected to the `/live` dashboard in QueueIQ-FE
- ⬜ Cross-frame tracking (ByteTrack) · Supabase persistence · ESP32 LED firmware
