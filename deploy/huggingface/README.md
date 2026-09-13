---
title: QueueIQ Vision API
emoji: 🛒
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Basket-aware checkout queue detection API (YOLOv8)
---

# QueueIQ Vision API

The inference backend for the QueueIQ smart-checkout dashboard. It detects the
shoppers in a queue with YOLOv8, finds their baskets and carts with YOLO-World,
rates how full each basket is with a MobileNetV2 classifier, and turns that into
a per-lane wait estimate that keeps learning from checkout feedback.

This Space serves the API only. The dashboard (`QueueIQ-FE`) is hosted
separately and calls this URL from the browser.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Model tier, loaded models, load error if any |
| GET | `/api/lanes` | Snapshot of every lane and the current recommendation |
| POST | `/api/lanes/{id}/analyze` | Analyze an uploaded frame (multipart `file`) into a lane |
| POST | `/api/analyze` | Stateless analysis, returns the annotated frame as base64 |
| GET | `/api/events` | Server-Sent Events stream used by the dashboard |
| GET | `/docs` | Interactive API documentation |

`/health` must report `"tier": "full"`. Anything else means the models did not
load, and `load_error` says why.

## Pointing the dashboard at this Space

Build the dashboard with the Space URL:

```bash
VITE_QUEUEIQ_API=https://<owner>-<space-name>.hf.space npm run build
```

or open an already-built dashboard once with `?api=https://<owner>-<space-name>.hf.space`.

## Things to know

- The Space must be **public**: the dashboard's live stream cannot send a token.
  Anyone with the URL can use the API.
- State lives in memory. A restart clears the lanes and the learned model
  parameters; the model warm-starts from synthetic data again.
- Free Spaces sleep after about 48 hours without visitors and take a minute to
  wake. Open the Space before a demo.

Built by DM Tech.
