# Deployment Guide — QueueIQ

QueueIQ is a **local appliance, not a cloud app**. Two processes run on one
machine (the demo laptop, or a mini-PC next to the checkout lanes):

| Process | Repository | Port | What it does |
|---|---|---|---|
| Vision API | `QueueIQ-AI` (`server.py`) | 8000 | YOLOv8 + basket detection + fullness classifier + wait prediction |
| Dashboard | `QueueIQ-FE` | 3000 (dev) / 4173 (production) | the `/live` control room |

The browser talks to the Vision API **directly** (fetch + Server-Sent Events),
so both must be reachable from whichever device shows the dashboard. The API
serves only loopback and private-LAN clients by default — model inference and
camera frames never leave the local network unless you deliberately open it up
(Path C).

Model weights are resolved from `QUEUEIQ_MODELS_DIR` before the checkout folder,
so on a server they can live outside git; `python fetch_models.py` puts them in
place and rebuilds what it can from public weights.

---

## Path A — demo laptop (recommended for the hackathon)

Everything on one machine, judges look at that screen. Two terminals.

**Prerequisites:** Python 3.11–3.13 and Node.js ≥ 22.13. Verified on
Python 3.13.1 / Node 22.23.2 / npm 10.9.8. Both repositories are private, so a
fresh clone needs a GitHub account with access (or an SSH deploy key, as in
Path C) — copying the folders across on a USB stick works just as well and is
the safer bet at a venue with no internet.

### Terminal 1 — Vision API

```bash
git clone https://github.com/D4V1DYL/QueueIQ-AI.git
cd QueueIQ-AI
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt -r requirements-server.txt
.venv/Scripts/python server.py
```

On Linux/macOS use `.venv/bin/pip` and `.venv/bin/python`. PyTorch is not
pinned to a CUDA build, so pip picks the right wheel on any platform and
inference runs at roughly 150–400 ms per frame on CPU — enough for the demo.
Only a Pascal GPU (GTX 10xx) needs an explicit CUDA build, because PyTorch
dropped Pascal support in cu128 and later:

```bash
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

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

## Path C — Oracle Cloud (or any public Linux server)

This works, but it is a different security posture from the LAN setup: the
dashboard becomes reachable by anyone who knows the address, and inference costs
CPU on your instance. Read the whole section before opening a port.

Three things have to be arranged, in this order.

### C1. Choose the instance

| Shape | Verdict |
|---|---|
| Ampere A1 (VM.Standard.A1.Flex, ARM, free tier up to 4 OCPU / 24 GB) | **Recommended.** PyTorch ships aarch64 wheels; plenty of headroom. |
| VM.Standard.E2.1.Micro (AMD, free tier, 1 GB RAM) | Tight but workable. The server measured **359 MB resident** after inference on this project's models. Add 1–2 GB of swap; a build or a second process will otherwise get killed. |

Disk: PyTorch alone unpacks to about **1.2 GB**, plus roughly 46 MB of weights.
A 50 GB boot volume is more than enough.

Nothing special is needed on ARM: `requirements.txt` does not pin a CUDA
build, and PyTorch publishes `manylinux_2_28_aarch64` wheels, so the normal
install works:

```bash
pip install -r requirements.txt -r requirements-server.txt
```

### C2. Get the code and the weights onto the server

The repositories are private, so the server needs its own read access: add a
**deploy key** (`ssh-keygen -t ed25519`, then paste the public key under the
repository's Settings → Deploy keys) and clone over SSH. A shallow clone keeps
the transfer small:

```bash
git clone --depth 1 git@github.com:D4V1DYL/QueueIQ-AI.git /opt/queueiq/api
git clone --depth 1 git@github.com:D4V1DYL/QueueIQ-FE.git /opt/queueiq/web
```

The weights are resolved from `QUEUEIQ_MODELS_DIR` before the checkout folder,
so they can live outside git — on a block volume, or simply in `/opt/queueiq/models`:

```bash
export QUEUEIQ_MODELS_DIR=/opt/queueiq/models
cd /opt/queueiq/api && python fetch_models.py
```

`fetch_models.py` rebuilds `yolov8n.pt` and `basket_world.pt` locally from public
Ultralytics weights, so only three files have to come from you —
`fullness_classifier.pt`, `class_names.txt` and (optionally) `basket_detector.pt`,
about 15 MB in total. Either copy them across once:

```bash
scp fullness_classifier.pt class_names.txt basket_detector.pt \
    opc@<instance-ip>:/opt/queueiq/models/
```

or upload them to an Object Storage bucket, create a **pre-authenticated
request** for the bucket (Object Storage → Bucket → Pre-Authenticated Requests →
"Enable object reads on the bucket"), and let the script pull them:

```bash
export QUEUEIQ_MODELS_URL=https://objectstorage.<region>.oraclecloud.com/p/<token>/n/<namespace>/b/queueiq-models/o
python fetch_models.py
```

Checksums for the trained files are baked into the script, so a truncated or
swapped download is rejected rather than silently loaded. Confirm with:

```bash
python fetch_models.py --check     # exits non-zero when tier full is impossible
```

### C3. Put both services behind one HTTPS origin

A dashboard served over HTTPS **cannot** call an HTTP API on port 8000 — browsers
block mixed content. One reverse proxy on port 443 serving both from the same
origin solves that, and it is also what makes the webcam capture work at all,
since `getUserMedia` requires a secure context.

The dashboard already expects this: over HTTPS it calls its own origin with no
port, so no configuration is needed as long as `/api/*` and `/health` reach the
Python service.

`/etc/caddy/Caddyfile`, with a DNS record pointing at the instance:

```caddyfile
queueiq.example.com {
    encode zstd gzip

    # Remove this block only if the demo is meant to be world-writable.
    basicauth {
        judge $2a$14$<bcrypt-hash-from-"caddy hash-password">
    }

    @api path /api/* /health
    handle @api {
        reverse_proxy 127.0.0.1:8000 {
            header_up X-Forwarded-For {remote_host}
            flush_interval -1          # required: /api/events is a live SSE stream
        }
    }

    handle {
        reverse_proxy 127.0.0.1:4173
    }
}
```

Caddy obtains and renews the certificate automatically. `flush_interval -1`
matters: without it the Server-Sent Events stream is buffered and the dashboard
never updates.

### C4. Open the network path — both layers

Oracle blocks inbound traffic twice, and forgetting the second layer is the most
common reason a correct deployment looks dead.

1. **Security list / NSG** in the console: add an ingress rule for TCP 443 (and
   80 for the certificate challenge) from `0.0.0.0/0`.
2. **The instance firewall**, which Oracle images enable by default:

```bash
# Oracle Linux / RHEL
sudo firewall-cmd --permanent --add-service=http --add-service=https
sudo firewall-cmd --reload

# Ubuntu images use iptables directly
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo netfilter-persistent save
```

Ports 8000 and 4173 stay closed to the internet — only Caddy talks to them.

### C5. Run both services under systemd

`/etc/systemd/system/queueiq-api.service`:

```ini
[Unit]
Description=QueueIQ Vision API
After=network-online.target

[Service]
User=queueiq
WorkingDirectory=/opt/queueiq/api
Environment=QUEUEIQ_MODELS_DIR=/opt/queueiq/models
Environment=QUEUEIQ_PUBLIC=1
Environment=QUEUEIQ_TRUST_PROXY=1
ExecStart=/opt/queueiq/api/.venv/bin/python server.py --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

`/etc/systemd/system/queueiq-web.service`:

```ini
[Unit]
Description=QueueIQ Dashboard
After=network-online.target

[Service]
User=queueiq
WorkingDirectory=/opt/queueiq/web
ExecStart=/usr/bin/npx vinext start -p 4173
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
cd /opt/queueiq/web && npm ci && npm run build
sudo systemctl daemon-reload
sudo systemctl enable --now queueiq-api queueiq-web caddy
journalctl -u queueiq-api -f        # confirm "tier=full"
```

`--host 127.0.0.1` is deliberate: the API listens only on loopback and is
reachable exclusively through Caddy.

### What the two new settings mean

`QUEUEIQ_PUBLIC=1` switches off the private-network guard. Without it every
request would be refused, because a public client is not a LAN client. The
server prints a warning at startup and logs one to the dashboard, since from
that moment authentication is entirely Caddy's job.

`QUEUEIQ_TRUST_PROXY=1` makes the server read the caller's address from
`X-Forwarded-For` instead of the socket. Behind a proxy every request appears to
come from `127.0.0.1`, which would make the LAN guard pass for the whole
internet — so if you ever turn `QUEUEIQ_PUBLIC` off again and rely on
`QUEUEIQ_ALLOWED_IPS`, this flag is what makes the allowlist meaningful. Only
set it when a proxy you control really does set that header.

### Limits worth knowing before you demo from a public URL

- **CPU inference is the bottleneck.** 150–400 ms per frame on this laptop; a
  free-tier instance will be slower, and every viewer who plays the virtual
  camera adds a frame every 2 seconds.
- **State is shared and in memory.** All viewers see the same four lanes, and
  anyone can press *Clear* or *Reset model*. There is one demo, not a session
  per visitor.
- **A restart forgets the lanes.** Only the learned model parameters survive, in
  `model_state.json` — put that file on the same volume as the models if you
  want it to persist across redeployments.
- **The videos and sample frames are third-party footage.** Fine for an internal
  demo, worth replacing with your own recordings before a public link is shared
  widely.

---

## Path D — a PaaS builder (Nixpacks: Coolify, Railway, Dokploy, Render)

Both repositories build with the stock Nixpacks detection, no Dockerfile and no
`nixpacks.toml`:

| Repository | Detected as | Install | Start |
|---|---|---|---|
| `QueueIQ-AI` | Python | `pip install -r requirements.txt` | `python server.py` |
| `QueueIQ-FE` | Node | `npm ci` + `npm run build` | `npm start` |

Three things make that work, and all three are already in the repositories:

- **`requirements.txt` pins no CUDA build.** A `+cu126` wheel only exists on
  PyTorch's own index for x86, so pinning one fails on ARM and on any builder
  that installs from PyPI. Plain `torch` resolves to the correct wheel.
- **Leave the install command at its default.** Nixpacks creates the virtualenv
  as part of its own install step; a custom `install_command` replaces that step,
  so the venv is never created and `pip` is not on the path.
- **`$PORT` is honoured.** Both services read the port the platform injects and
  bind `0.0.0.0` when it is present, so no start-command override is needed.
  `requirements-server.txt` uses `opencv-python-headless`, which does not need
  the `libGL` system library that slim images lack.

Set these on the API service, since a PaaS reaches it through its own proxy:

```
QUEUEIQ_PUBLIC=1
QUEUEIQ_TRUST_PROXY=1
QUEUEIQ_MODELS_DIR=/data/models      # optional, if you mount a volume
```

Everything in Path C about HTTPS still applies: the dashboard and the API have to
answer on **one hostname**, with `/api/*` and `/health` routed to the Python
service, or the browser blocks the calls as mixed content. Most PaaS front-ends
can do that with a path rule; if yours cannot, deploy the dashboard and put the
API on the same hostname behind the Caddy configuration above.

The weights ship in the repository, so a plain build already reaches
`tier=full`. Mount a volume and run `python fetch_models.py` only if you would
rather keep them out of the image.

---

## Configuration reference

| Variable | Default | Meaning |
|---|---|---|
| `QUEUEIQ_HOST` | `127.0.0.1` | Bind address (`--host` overrides) |
| `QUEUEIQ_PORT` | `8000` | Port (`--port` overrides) |
| `QUEUEIQ_MODE` | `auto` | `mock` runs the UI with placeholder detections and no torch |
| `QUEUEIQ_ALLOWED_IPS` | empty | Comma-separated allowlist; empty means any private-LAN client |
| `QUEUEIQ_CORS` | `*` | Comma-separated allowed browser origins |
| `QUEUEIQ_PUBLIC` | unset | `1` disables the private-network guard. Only behind an authenticating proxy — see Path C |
| `QUEUEIQ_TRUST_PROXY` | unset | `1` reads the caller's address from `X-Forwarded-For` instead of the socket |
| `QUEUEIQ_MODELS_DIR` | the repo folder | Where the weights are loaded from; falls back to the repo per file |
| `QUEUEIQ_MODELS_URL` | unset | Base URL `fetch_models.py` downloads the trained weights from |
| `PORT` | unset | PaaS convention; when set, the server uses it and binds `0.0.0.0` |
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
| Public deployment: every request is `403 LAN-only` | `QUEUEIQ_PUBLIC=1` is not set on the service. |
| Public deployment: page loads, lanes never update | The proxy is buffering Server-Sent Events. Caddy needs `flush_interval -1`, nginx needs `proxy_buffering off`. |
| Public deployment: nothing answers on 443 | Two firewalls. Check the OCI security list **and** `firewall-cmd --list-all` / `iptables -L INPUT` on the instance. |
| `python fetch_models.py` reports MISSING | The three trained files are not public. Copy them with scp or serve them through `QUEUEIQ_MODELS_URL`. |
| Nixpacks build fails: `pip: command not found` | A custom `install_command` was set. Nixpacks creates the virtualenv in its own install step, so leave that field empty. |
| Nixpacks build fails resolving `torch` | An old checkout still pins `torch==…+cu126`. Those wheels exist only on PyTorch's x86 index; pull the current `requirements.txt`. |
| `ImportError: libGL.so.1` | An old `requirements-server.txt` with `opencv-python`. The current one uses `opencv-python-headless`. |
