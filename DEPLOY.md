# Deployment Guide — QueueIQ

QueueIQ is a **local appliance, not a cloud app**. Two processes run on one
machine (the demo laptop, or a mini-PC next to the checkout lanes):

| Process | Repository | Port | What it does |
|---|---|---|---|
| Vision API | `QueueIQ-AI` (`server.py`) | 8000 | YOLOv8 + basket detection + fullness classifier + wait prediction |
| Dashboard | `QueueIQ-FE` | 3000 (dev) / 4173 (production) | the `/live` control room |

The browser talks to the Vision API **directly** (fetch + Server-Sent Events),
so both must be reachable from whichever device shows the dashboard. The API
serves only loopback and private-LAN clients by design — model inference and
camera frames never leave the local network.

---

## Path A — demo laptop (recommended for the hackathon)

Everything on one machine, judges look at that screen. Two terminals.

**Prerequisites:** Python 3.11–3.13 and Node.js ≥ 22.13. Verified on
Python 3.13.1 / Node 22.23.2 / npm 10.9.8.

### Terminal 1 — Vision API

```bash
git clone https://github.com/D4V1DYL/QueueIQ-AI.git
cd QueueIQ-AI
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/pip install -r requirements-server.txt
.venv/Scripts/python server.py
```

On Linux/macOS use `.venv/bin/pip` and `.venv/bin/python`. The `cu126` index is
only needed for Pascal GPUs (GTX 10xx); on a CPU-only machine plain
`pip install -r requirements.txt` is fine — inference runs at roughly
150–400 ms per frame on CPU, which is enough for the demo.

All model weights (`yolov8n.pt`, `basket_world.pt`, `fullness_classifier.pt`,
`class_names.txt`, `basket_detector.pt`) and the two demo videos ship inside the
repository, so nothing is downloaded at startup. Confirm the startup line reads:

```
QueueIQ Vision API -> http://127.0.0.1:8000  (tier=full)
```

`tier=full` is what you want. See the troubleshooting table if it says
`heuristic` or `mock`.

### Terminal 2 — Dashboard

```bash
git clone https://github.com/D4V1DYL/QueueIQ-FE.git
cd QueueIQ-FE
npm install
npm run build
npx vinext start -p 4173
```

Open **http://localhost:4173/live**. The header pill must read
`LIVE AI · YOLO + CLASSIFIER`.

`npm run dev` also works and is fine for a demo, but the production build starts
faster and has no dev-server overhead.

---

## Path B — LAN (phones, a second laptop, the judges' own devices)

Same two commands, with the API bound to every interface:

```bash
# Terminal 1
.venv/Scripts/python server.py --host 0.0.0.0

# Terminal 2 (vinext already binds 0.0.0.0)
npx vinext start -p 4173
```

Find the laptop's LAN address (`ipconfig` on Windows, `ip a` on Linux), then
open **http://&lt;LAN-IP&gt;:4173/live** on the other device — for example
`http://192.168.1.3:4173/live`.

No configuration is needed: the dashboard derives the API address from the host
that served the page, so a page loaded from `192.168.1.3:4173` calls
`192.168.1.3:8000` automatically.

**Windows Firewall blocks inbound connections by default.** Either accept the
prompt Windows shows the first time, or open the two ports once from an
Administrator terminal:

```powershell
netsh advfirewall firewall add rule name="QueueIQ API" dir=in action=allow protocol=TCP localport=8000 profile=private
netsh advfirewall firewall add rule name="QueueIQ Dashboard" dir=in action=allow protocol=TCP localport=4173 profile=private
```

Use `profile=private` and make sure the Wi-Fi is marked as a private network.
Remove the rules afterwards with `netsh advfirewall firewall delete rule name="QueueIQ API"`.

### Pointing the dashboard at a different machine

If the API runs on another host than the dashboard, append `?api=` once — it is
remembered in that browser:

```
http://192.168.1.3:4173/live?api=192.168.1.50:8000
```

The same field appears in the "Vision server not reachable" panel whenever the
dashboard cannot connect.

### Tightening access

```bash
QUEUEIQ_ALLOWED_IPS=192.168.1.20,192.168.1.55 .venv/Scripts/python server.py --host 0.0.0.0
```

Only those addresses (plus loopback) are served. `QUEUEIQ_CORS` narrows the
allowed browser origins the same way.

---

## Path C — public hosting

**Not supported as-is, and not recommended for the hackathon.** Two things break:

1. The Vision API refuses any client that is not loopback or a private LAN
   address. That guard is deliberate.
2. A dashboard served over HTTPS cannot call an HTTP API — browsers block mixed
   content, and the models cannot run on a static host anyway.

If a public demo is genuinely required, the shape that works is: keep the Vision
API on the local machine, expose it through an HTTPS tunnel
(`cloudflared tunnel --url http://localhost:8000`), remove the LAN guard in
`server.py`, put real authentication in front of it, then deploy the dashboard
(`npm run build`, `dist/` is a Cloudflare Workers bundle — `npm start` runs it
through Wrangler) and point it at the tunnel URL with `?api=`. That is a
different security posture from the one this project was built for, so treat it
as a separate piece of work rather than a deployment step.

---

## Configuration reference

| Variable | Default | Meaning |
|---|---|---|
| `QUEUEIQ_HOST` | `127.0.0.1` | Bind address (`--host` overrides) |
| `QUEUEIQ_PORT` | `8000` | Port (`--port` overrides) |
| `QUEUEIQ_MODE` | `auto` | `mock` runs the UI with placeholder detections and no torch |
| `QUEUEIQ_ALLOWED_IPS` | empty | Comma-separated allowlist; empty means any private-LAN client |
| `QUEUEIQ_CORS` | `*` | Comma-separated allowed browser origins |
| `YOLO_OFFLINE` | set to `1` by `server.py` | Stops Ultralytics contacting the network |

The dashboard has no build-time configuration. The API address comes from the
page host, the `?api=` parameter, or the reconnect panel.

---

## Demo-day checklist

1. Both repositories cloned and dependencies installed **before** the venue,
   including `npm install` and `npm run build`.
2. `python server.py` prints `tier=full`.
3. `http://localhost:4173/live` shows `LIVE AI · YOLO + CLASSIFIER`.
4. Judge demo buttons work: *Seed lanes* → *Full-basket surge* (a lane flips to
   red and the recommendation moves) → *Sample CCTV frame*.
5. *Sample footage* plays a video as a virtual camera and the annotated frame
   updates every 2 seconds.
6. If demoing from a phone: firewall rules added, the LAN URL opens, and the
   laptop is on the same Wi-Fi (not a guest network with client isolation).
7. Optional reset before the judges arrive:
   `curl -X POST localhost:8000/api/model/reset -H "content-type: application/json" -d "{\"warm_start\":true}"`
   and the *Clear* button on the dashboard.

Nothing requires internet access at demo time. If the venue Wi-Fi is unusable,
a phone hotspot or a direct Ethernet cable between two laptops works, and the
laptop-only path needs no network at all.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Header says `MOCK INFERENCE` | torch/ultralytics not installed in the interpreter running `server.py`. Install `requirements.txt` into the same venv. |
| Header says `LIVE AI · YOLO + CV HEURISTIC` | `fullness_classifier.pt` or `class_names.txt` missing from the `QueueIQ-AI` folder. They ship with the repository; re-clone or `git checkout` them. |
| Header says `Vision server offline` | The API is not running, or the page is pointed at the wrong host. Check the address printed under the "not reachable" panel and use the field there. |
| Works on the laptop, not from a phone | Firewall, or the API was started without `--host 0.0.0.0`. Confirm with `curl http://<LAN-IP>:8000/health` from the phone's browser. |
| Phone shows the page but no lanes | The phone reached the dashboard but not the API. Both must be allowed through the firewall, not just port 4173. |
| `ERR_CONNECTION_REFUSED` on port 8000 | The API crashed or the port is taken. `netstat -ano \| findstr :8000`. |
| Inference feels slow | Expected on CPU (150–400 ms/frame). Reduce the virtual camera rate, or run on a machine with a supported GPU. |
| Model parameters look wrong after testing | `POST /api/model/reset` with `{"warm_start": true}` restores the synthetic warm start. |
