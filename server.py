"""
server.py — QueueIQ Vision API

Bridge between the AI pipeline (queueiq_engine.py) and the Next.js dashboard
(QueueIQ-FE, page /live). One Python process, no database: lane state lives in
memory, the online-learning model parameters are persisted to model_state.json.

    python server.py                       # auto: use whatever models are present
    python server.py --mode mock           # no torch, UI demo only
    python server.py --host 0.0.0.0        # expose on the LAN (phone/ESP32 on the same Wi-Fi)

Security (local-backend policy): only loopback / private-LAN clients are
served. Restrict further with
    QUEUEIQ_ALLOWED_IPS=192.168.1.20,192.168.1.55

Main endpoints (JSON unless stated otherwise):
    GET  /health                         model tier, device, version
    GET  /api/lanes                      snapshot of all lanes + recommendation
    POST /api/lanes/{id}/analyze         upload a frame (multipart 'file') OR
                                         form 'example=<name>' -> detections
    GET  /api/lanes/{id}/frame.jpg       latest annotated frame
    POST /api/lanes/{id}/open            {"open": true|false}
    POST /api/lanes/{id}/complete        {"actual_sec": 87} -> feedback ->
                                         the model learns (SGD step)
    POST /api/analyze                    stateless: frame -> result + b64 image
    GET  /api/model                      parameters + accuracy history
    POST /api/model/reset                {"warm_start": true}
    GET  /api/examples                   sample frames bundled with the repo
    POST /api/demo/scenario              {"name": "seed"|"surge"|"cctv"|"clear"}
    GET  /api/videos                     demo footage in videos/ (virtual camera)
    POST /api/lanes/{id}/video           {"name": "x.mp4", "interval_sec": 2} ->
                                         play the video through the pipeline into a lane
    POST /api/lanes/{id}/video/stop
    GET  /api/led            /api/led/{id}   light colour (plain text, for an ESP32)
    GET  /api/events                     Server-Sent Events: lanes/log/model
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import ipaddress
import io
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Offline mode: the weights live in this folder, so never let Ultralytics
# try to check for updates or download anything during an offline demo.
os.environ.setdefault("YOLO_OFFLINE", "1")

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from PIL import Image
from pydantic import BaseModel

from queueiq_engine import (
    FULLNESS_TO_ITEMS, THRESHOLD_AMBER, THRESHOLD_GREEN, QueueEngine,
    OnlineRegressor, parse_zone, regressor_from_disk, status_for,
)

VERSION = "0.3.0"
VENDOR = "DM Tech"
ROOT = Path(__file__).resolve().parent
EXAMPLES_DIR = ROOT / "examples"
VIDEOS_DIR = ROOT / "videos"          # demo footage -> virtual camera
VIDEO_EXT = (".mp4", ".mov", ".webm", ".mkv", ".avi")
LANE_NAMES = {1: "Lane 01", 2: "Lane 02", 3: "Lane 03", 4: "Lane 04"}
MIN_MEASURED_FEEDBACK_SEC = 15.0   # Done under 15 s = not a real checkout, do not learn



# =====================================================================
# Request bodies (module-level: FastAPI must resolve them by name because
# of `from __future__ import annotations`)
# =====================================================================
class OpenBody(BaseModel):
    open: bool


class CompleteBody(BaseModel):
    actual_sec: Optional[float] = None


class ResetBody(BaseModel):
    warm_start: bool = True


class ScenarioBody(BaseModel):
    name: str


class VideoBody(BaseModel):
    name: str
    interval_sec: float = 2.0     # analyse 1 frame every N seconds of video and wall time
    loop: bool = True


# =====================================================================
# State
# =====================================================================
@dataclass
class Shopper:
    id: str
    fullness: str
    confidence: float
    est_items: int
    est_sec: float
    source: str                 # basket | carry-region | mock | scenario
    detected_at: float
    person_box: Optional[list] = None
    crop_box: Optional[list] = None
    detail: dict = field(default_factory=dict)


@dataclass
class Lane:
    id: int
    name: str
    open: bool = True
    shoppers: list[Shopper] = field(default_factory=list)
    zone: Optional[list] = None
    frame: Optional[bytes] = None
    frame_version: int = 0
    last_result: Optional[dict] = None
    last_analyzed_at: Optional[float] = None
    front_started_at: Optional[float] = None
    completed: int = 0
    video_name: Optional[str] = None
    video_task: Optional[asyncio.Task] = None
    video_token: object = None      # identity of the active playback (safe cleanup)

    def remaining_of(self, i: int, now: float) -> float:
        s = self.shoppers[i]
        if i == 0 and self.front_started_at:
            return max(0.0, s.est_sec - (now - self.front_started_at))
        return s.est_sec

    def wait_sec(self, now: float) -> float:
        return sum(self.remaining_of(i, now) for i in range(len(self.shoppers)))

    def signal(self, now: float) -> str:
        return "closed" if not self.open else status_for(self.wait_sec(now))

    def to_dict(self, now: float) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "open": self.open,
            "wait_sec": round(self.wait_sec(now), 1),
            "signal": self.signal(now),
            "shoppers": [
                {
                    "id": s.id, "fullness": s.fullness, "confidence": s.confidence,
                    "est_items": s.est_items, "est_sec": round(s.est_sec, 1),
                    "remaining_sec": round(self.remaining_of(i, now), 1),
                    "source": s.source, "person_box": s.person_box,
                    "crop_box": s.crop_box, "detail": s.detail,
                    "position": i + 1,
                }
                for i, s in enumerate(self.shoppers)
            ],
            "n_shoppers": len(self.shoppers),
            "total_items": sum(s.est_items for s in self.shoppers),
            "avg_fullness": _avg_fullness(self.shoppers),
            "root_cause": _root_cause(self, now),
            "last_analyzed_at": self.last_analyzed_at,
            "front_started_at": self.front_started_at,
            "frame_url": f"/api/lanes/{self.id}/frame.jpg?v={self.frame_version}" if self.frame else None,
            "n_detected_total": (self.last_result or {}).get("n_detected_total"),
            "inference_ms": (self.last_result or {}).get("inference_ms"),
            "completed": self.completed,
            "video_source": self.video_name,
        }


FULLNESS_SCORE = {"empty": 0, "light": 1, "medium": 2, "full": 3, "no_basket_with_items": 1}


def _avg_fullness(shoppers: list[Shopper]) -> Optional[float]:
    if not shoppers:
        return None
    return round(sum(FULLNESS_SCORE.get(s.fullness, 1) for s in shoppers) / len(shoppers), 2)


def _root_cause(lane: Lane, now: float) -> str:
    """Short explanation for the manager dashboard: why this lane is
    slow/fast (the differentiator: not just a headcount)."""
    if not lane.open:
        return "Lane closed."
    n = len(lane.shoppers)
    if n == 0:
        return "Empty lane — ready for the next shopper."
    full = sum(1 for s in lane.shoppers if s.fullness == "full")
    avg = _avg_fullness(lane.shoppers) or 0
    sig = lane.signal(now)
    if sig == "green":
        return f"{n} shopper{'s' if n > 1 else ''} with light baskets — low workload."
    if full >= max(1, n // 2):
        return f"HIGH BASKET LOAD: {full} of {n} shoppers carry full baskets."
    if n >= 4:
        return f"HIGH CUSTOMER COUNT: {n} shoppers queued, mostly small baskets."
    if avg >= 2:
        return f"Medium-to-full baskets across {n} shoppers raise the workload."
    return f"{n} shoppers with mixed basket sizes."


class Broadcaster:
    def __init__(self):
        self.clients: set[asyncio.Queue] = set()

    async def publish(self, event: str, data: dict):
        payload = f"event: {event}\ndata: {json.dumps(data)}\n\n"
        for q in list(self.clients):
            if q.qsize() > 200:
                self.clients.discard(q)
                continue
            q.put_nowait(payload)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self.clients.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self.clients.discard(q)


class Store:
    def __init__(self, engine: QueueEngine):
        self.engine = engine
        self.lanes: dict[int, Lane] = {i: Lane(i, n, open=(i != 4)) for i, n in LANE_NAMES.items()}
        self.log: list[dict] = []
        self.bus = Broadcaster()
        self.started_at = time.time()
        self._seq = 0
        self._beat = 0

    # --- helpers ---
    def lane(self, lane_id: int) -> Lane:
        lane = self.lanes.get(lane_id)
        if not lane:
            raise HTTPException(404, f"Lane {lane_id} not found")
        return lane

    def snapshot(self) -> dict:
        now = time.time()
        lanes = [l.to_dict(now) for l in self.lanes.values()]
        open_lanes = [l for l in lanes if l["open"]]
        best = min(open_lanes, key=lambda l: (l["wait_sec"], l["id"])) if open_lanes else None
        return {
            "ts": now,
            "tier": self.engine.tier,
            "lanes": lanes,
            "recommendation": {
                "lane_id": best["id"] if best else None,
                "lane_name": best["name"] if best else None,
                "wait_sec": best["wait_sec"] if best else None,
                "reason": (best["root_cause"] if best else "No open lanes."),
            },
            "totals": {
                "shoppers": sum(l["n_shoppers"] for l in lanes),
                "items": sum(l["total_items"] for l in lanes),
                "completed": sum(l["completed"] for l in lanes),
            },
        }

    async def add_log(self, text: str, kind: str = "info", lane_id: Optional[int] = None):
        self._seq += 1
        rec = {"id": self._seq, "text": text, "kind": kind, "lane_id": lane_id, "ts": time.time()}
        self.log.insert(0, rec)
        del self.log[60:]
        await self.bus.publish("log", rec)

    async def push_lanes(self):
        await self.bus.publish("lanes", self.snapshot())

    async def push_model(self):
        await self.bus.publish("model", self.engine.regressor.summary())

    # --- core actions ---
    async def analyze_into_lane(self, lane: Lane, img: Image.Image, origin: str) -> dict:
        result, jpeg = await asyncio.to_thread(self.engine.analyze, img, lane.zone, True)
        now = time.time()
        had_front = bool(lane.shoppers)
        lane.shoppers = [
            Shopper(
                id=f"{lane.id}-{uuid.uuid4().hex[:4].upper()}",
                fullness=d["fullness"], confidence=d["confidence"],
                est_items=d["est_items"], est_sec=d["est_sec"], source=d["source"],
                detected_at=now, person_box=d["person_box"], crop_box=d["crop_box"],
                detail=d.get("detail") or {},
            )
            for d in result["detections"]
        ]
        # the same front shopper is still being served -> keep the timer
        lane.front_started_at = (lane.front_started_at if had_front and lane.shoppers else
                                 (now if lane.shoppers else None))
        lane.frame = jpeg
        lane.frame_version += 1
        lane.last_result = result
        lane.last_analyzed_at = now
        sig = lane.signal(now)
        await self.add_log(
            f"{lane.name}: {result['n_person']} shopper(s) detected from {origin} "
            f"→ ~{result['total_sec']:.0f}s wait → {sig.upper()} "
            f"[{result['tier']}, {result['inference_ms']:.0f} ms]",
            kind="detect", lane_id=lane.id)
        await self.push_lanes()
        return result

    async def complete_front(self, lane: Lane, actual_sec: Optional[float]) -> dict:
        if not lane.shoppers:
            raise HTTPException(409, f"{lane.name} has no shopper to complete")
        now = time.time()
        s = lane.shoppers.pop(0)
        measured = (now - lane.front_started_at) if lane.front_started_at else None
        actual = actual_sec if actual_sec is not None else measured
        lane.front_started_at = now if lane.shoppers else None
        lane.completed += 1
        learned = None
        skipped_reason = None
        # Guard: a MEASURED time (Done button) that is too short is almost
        # certainly not a real checkout (an operator clicking during a demo) ->
        # do not learn from it. Explicit numbers ("Teach the model") are accepted.
        if actual_sec is None and (measured is None or measured < MIN_MEASURED_FEEDBACK_SEC):
            skipped_reason = (f"measured {measured:.0f}s is under {MIN_MEASURED_FEEDBACK_SEC}s — "
                              f"not a real checkout, model left unchanged"
                              if measured is not None else "no timing available")
            actual = None
        if actual is not None and actual >= 3:
            learned = self.engine.regressor.update(s.est_items, float(actual), source="live")
            self.engine.regressor.save()
            await self.add_log(
                f"{lane.name}: {s.id} ({s.fullness}, ~{s.est_items} items) checked out in "
                f"{actual:.0f}s vs predicted {learned['predicted_sec']:.0f}s → model updated "
                f"({learned['intercept_before']:.1f}+{learned['slope_before']:.2f}x → "
                f"{learned['intercept']:.1f}+{learned['slope']:.2f}x)",
                kind="learn", lane_id=lane.id)
            await self.push_model()
        else:
            await self.add_log(f"{lane.name}: {s.id} checked out — "
                               f"{skipped_reason or 'no timing feedback'}.",
                               kind="info", lane_id=lane.id)
        await self.push_lanes()
        return {"shopper": s.__dict__, "actual_sec": actual, "measured_sec": measured,
                "learned": learned, "skipped_reason": skipped_reason}

    def add_scenario_shopper(self, lane: Lane, fullness: str, conf: float = 0.9):
        items = FULLNESS_TO_ITEMS[fullness]
        now = time.time()
        lane.shoppers.append(Shopper(
            id=f"{lane.id}-{uuid.uuid4().hex[:4].upper()}", fullness=fullness, confidence=conf,
            est_items=items, est_sec=round(self.engine.regressor.predict(items), 1),
            source="scenario", detected_at=now))
        if lane.front_started_at is None:
            lane.front_started_at = now

    # --- virtual camera: play a video file through the pipeline ---
    async def play_video(self, lane: Lane, path: Path, interval: float, loop: bool):
        """Read one frame every `interval` seconds (video time == wall time) and
        analyse it into the lane as if it came from the overhead camera."""
        import cv2  # lazy: only needed for this feature

        token = object()
        lane.video_token = token
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            await self.add_log(f"{lane.name}: cannot open video {path.name}", kind="system", lane_id=lane.id)
            lane.video_name, lane.video_task = None, None
            await self.push_lanes()
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(fps * interval)))
        await self.add_log(f"{lane.name}: virtual camera started — {path.name} "
                           f"({total / fps:.0f}s, 1 frame every {interval:g}s)", kind="system", lane_id=lane.id)
        idx = 0
        try:
            while True:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok:
                    if not loop or total == 0:
                        break
                    idx = 0
                    continue
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                await self.analyze_into_lane(lane, img, f"video:{path.name}@{idx / fps:.0f}s")
                idx += step
                if total and idx >= total:
                    if not loop:
                        break
                    idx = 0
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
        finally:
            cap.release()
            if lane.video_token is token:   # not superseded by another playback
                lane.video_name, lane.video_task, lane.video_token = None, None, None
                await self.add_log(f"{lane.name}: virtual camera stopped.", kind="system", lane_id=lane.id)
                await self.push_lanes()

    def stop_video(self, lane: Lane):
        """Cancel the playback task; cleanup and logging happen in play_video's finally."""
        if lane.video_task and not lane.video_task.done():
            lane.video_task.cancel()

    async def tick(self):
        """1-second loop: countdown and auto-advance when the front shopper is done
        (no feedback to the model — only real feedback is used for learning)."""
        while True:
            await asyncio.sleep(1)
            now = time.time()
            changed = False
            for lane in self.lanes.values():
                if lane.shoppers and lane.front_started_at and lane.remaining_of(0, now) <= 0:
                    s = lane.shoppers.pop(0)
                    lane.completed += 1
                    lane.front_started_at = now if lane.shoppers else None
                    await self.add_log(f"{lane.name}: {s.id} finished checkout (auto-advanced, "
                                       f"no timing feedback).", kind="info", lane_id=lane.id)
                    changed = True
            self._beat += 1
            # The countdown runs client-side (front_started_at + est_sec);
            # the server only pushes on changes plus a heartbeat every 10 seconds.
            if changed or self._beat % 10 == 0:
                await self.push_lanes()


# =====================================================================
# App
# =====================================================================
def create_app(mode: str = "auto") -> FastAPI:
    engine = QueueEngine(mode=mode, regressor=regressor_from_disk()).load()
    store = Store(engine)
    app = FastAPI(title="QueueIQ Vision API", version=VERSION)
    app.state.store = store

    allowed_env = os.environ.get("QUEUEIQ_ALLOWED_IPS", "").strip()
    allowed_ips = {ip.strip() for ip in allowed_env.split(",") if ip.strip()}
    public = os.environ.get("QUEUEIQ_PUBLIC", "").strip().lower() in ("1", "true", "yes")
    trust_proxy = os.environ.get("QUEUEIQ_TRUST_PROXY", "").strip().lower() in ("1", "true", "yes")

    def client_ip(request: Request) -> str:
        """The caller's address. Behind a reverse proxy every request appears to
        come from the proxy itself, so the real address is only known when the
        deployment explicitly says the X-Forwarded-For header can be trusted."""
        if trust_proxy:
            fwd = request.headers.get("x-forwarded-for", "")
            if fwd:
                return fwd.split(",")[0].strip()
        return request.client.host if request.client else "127.0.0.1"

    @app.middleware("http")
    async def lan_only(request: Request, call_next):
        if public:
            return await call_next(request)
        host = client_ip(request)
        try:
            ip = ipaddress.ip_address(host)
            ok = ip.is_loopback or ip.is_private or ip.is_link_local
        except ValueError:
            ok = host in ("localhost", "testclient")
        if allowed_ips:
            ok = host in allowed_ips or (ok and ipaddress.ip_address(host).is_loopback)
        if not ok:
            return JSONResponse({"detail": "QueueIQ API is LAN-only."}, status_code=403)
        return await call_next(request)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in os.environ.get("QUEUEIQ_CORS", "*").split(",")],
        allow_methods=["*"], allow_headers=["*"],
    )

    @app.on_event("startup")
    async def _startup():
        asyncio.create_task(store.tick())
        await store.add_log(
            f"QueueIQ Vision API {VERSION} online — tier={engine.tier}, device={engine.device}, "
            f"model {engine.regressor.summary()['formula']} "
            f"({engine.regressor.summary()['n_updates']} feedback samples).", kind="system")
        if public:
            await store.add_log(
                "QUEUEIQ_PUBLIC is set: the private-network guard is OFF. Anyone who can "
                "reach this port can drive the demo and spend CPU on inference — put "
                "authentication in front of it (see DEPLOY.md).", kind="system")

    # ---------------- health ----------------
    @app.get("/health")
    async def health():
        info = engine.info()
        return {
            "ok": True, "service": "QueueIQ Vision API", "version": VERSION, "vendor": VENDOR,
            "uptime_sec": round(time.time() - store.started_at, 1),
            "sse_clients": len(store.bus.clients),
            **info,
            "model": {k: v for k, v in engine.regressor.summary().items() if k != "history"},
        }

    # ---------------- lanes ----------------
    @app.get("/api/lanes")
    async def get_lanes():
        return store.snapshot()

    @app.get("/api/recommendation")
    async def get_recommendation():
        return store.snapshot()["recommendation"]

    @app.get("/api/lanes/{lane_id}")
    async def get_lane(lane_id: int):
        return store.lane(lane_id).to_dict(time.time())

    @app.get("/api/lanes/{lane_id}/frame.jpg")
    async def get_frame(lane_id: int):
        lane = store.lane(lane_id)
        if not lane.frame:
            raise HTTPException(404, "No frame analyzed for this lane yet")
        return Response(lane.frame, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    async def _read_image(file: Optional[UploadFile], example: Optional[str]) -> tuple[Image.Image, str]:
        if file is not None and file.filename:
            data = await file.read()
            if len(data) > 15 * 1024 * 1024:
                raise HTTPException(413, "Image larger than 15 MB")
            try:
                return Image.open(io.BytesIO(data)), f"upload:{file.filename}"
            except Exception:
                raise HTTPException(400, "File is not a readable image")
        if example:
            p = (EXAMPLES_DIR / Path(example).name)
            if not p.exists() or p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                raise HTTPException(404, f"Example '{example}' not found")
            return Image.open(p), f"example:{p.name}"
        raise HTTPException(400, "Send an image as multipart 'file' or form field 'example'")

    @app.post("/api/lanes/{lane_id}/analyze")
    async def analyze_lane(lane_id: int, file: Optional[UploadFile] = File(None),
                           example: Optional[str] = Form(None), zone: Optional[str] = Form(None)):
        lane = store.lane(lane_id)
        if not lane.open:
            raise HTTPException(409, f"{lane.name} is closed")
        if zone:
            try:
                lane.zone = parse_zone(zone)
            except Exception as e:
                raise HTTPException(400, f"Bad zone: {e}")
        img, origin = await _read_image(file, example)
        result = await store.analyze_into_lane(lane, img, origin)
        return {"result": result, "lane": lane.to_dict(time.time()),
                "recommendation": store.snapshot()["recommendation"]}

    @app.post("/api/analyze")
    async def analyze_stateless(file: Optional[UploadFile] = File(None),
                                example: Optional[str] = Form(None), zone: Optional[str] = Form(None)):
        img, origin = await _read_image(file, example)
        z = parse_zone(zone) if zone else None
        result, jpeg = await asyncio.to_thread(engine.analyze, img, z, True)
        result["origin"] = origin
        result["annotated_jpeg_b64"] = base64.b64encode(jpeg).decode() if jpeg else None
        return result

    @app.post("/api/lanes/{lane_id}/open")
    async def set_open(lane_id: int, body: OpenBody):
        lane = store.lane(lane_id)
        if not body.open and lane.shoppers:
            raise HTTPException(409, "A lane can only close after its queue is empty")
        lane.open = body.open
        await store.add_log(f"{lane.name} {'opened' if body.open else 'closed'} by operator.",
                            kind="info", lane_id=lane.id)
        await store.push_lanes()
        return lane.to_dict(time.time())

    @app.post("/api/lanes/{lane_id}/complete")
    async def complete(lane_id: int, body: CompleteBody = CompleteBody()):
        lane = store.lane(lane_id)
        if body.actual_sec is not None and not (1 <= body.actual_sec <= 3600):
            raise HTTPException(400, "actual_sec must be between 1 and 3600")
        out = await store.complete_front(lane, body.actual_sec)
        out["lane"] = lane.to_dict(time.time())
        return out

    # ---------------- model ----------------
    @app.get("/api/model")
    async def get_model():
        return engine.regressor.summary()

    @app.post("/api/model/reset")
    async def reset_model(body: ResetBody = ResetBody()):
        engine.regressor = OnlineRegressor()
        n = engine.regressor.warm_start() if body.warm_start else 0
        engine.regressor.save()
        await store.add_log(f"Prediction model reset ({'warm start from ' + str(n) + ' synthetic transactions' if n else 'cold start: 10 + 4x item'}).", kind="system")
        await store.push_model()
        await store.push_lanes()
        return engine.regressor.summary()

    # ---------------- examples & demo ----------------
    @app.get("/api/examples")
    async def examples():
        return [p.name for p in sorted(EXAMPLES_DIR.glob("*"))
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp") and "_annotated" not in p.name]

    @app.get("/api/examples/{name}")
    async def example_file(name: str):
        p = EXAMPLES_DIR / Path(name).name
        if not p.exists():
            raise HTTPException(404)
        return Response(p.read_bytes(), media_type="image/jpeg")

    @app.post("/api/demo/scenario")
    async def scenario(body: ScenarioBody):
        name = body.name
        if name == "clear":
            for l in store.lanes.values():
                store.stop_video(l)
                l.shoppers, l.front_started_at, l.frame, l.last_result = [], None, None, None
            await store.add_log("All lanes cleared.", kind="system")
        elif name == "seed":
            plan = {1: ["full", "full"], 2: ["light", "no_basket_with_items", "no_basket_with_items"],
                    3: ["medium", "medium"], 4: []}
            for lid, labels in plan.items():
                l = store.lanes[lid]
                l.shoppers, l.front_started_at = [], None
                for lab in labels:
                    store.add_scenario_shopper(l, lab)
            await store.add_log("Scenario SEED: 2 full baskets (Lane 01) vs 3 light/hand-carried (Lane 02) "
                                "vs 2 medium (Lane 03). More people ≠ slower.", kind="system")
        elif name == "surge":
            snap = store.snapshot()
            lid = snap["recommendation"]["lane_id"]
            if lid is None:
                raise HTTPException(409, "No open lane")
            l = store.lanes[lid]
            before = l.signal(time.time())
            store.add_scenario_shopper(l, "full", 0.93)
            after = l.signal(time.time())
            await store.add_log(f"Scenario SURGE: a full-basket shopper joins {l.name} → "
                                f"{before.upper()} → {after.upper()}. Recommendation re-evaluated.",
                                kind="system", lane_id=lid)
        elif name == "cctv":
            names = await examples()
            if not names:
                raise HTTPException(404, "No example frames in examples/")
            l = store.lanes[1]
            l.open = True
            await store.analyze_into_lane(l, Image.open(EXAMPLES_DIR / names[0]), f"example:{names[0]}")
        else:
            raise HTTPException(400, "Unknown scenario: seed | surge | cctv | clear")
        await store.push_lanes()
        return store.snapshot()

    # ---------------- video as a virtual camera ----------------
    def _list_videos():
        if not VIDEOS_DIR.is_dir():
            return []
        return [{"name": p.name, "size_mb": round(p.stat().st_size / 1e6, 1)}
                for p in sorted(VIDEOS_DIR.iterdir()) if p.suffix.lower() in VIDEO_EXT]

    @app.get("/api/videos")
    async def videos():
        return _list_videos()

    @app.get("/api/videos/{name}")
    async def video_file(name: str):
        p = VIDEOS_DIR / Path(name).name
        if not p.exists() or p.suffix.lower() not in VIDEO_EXT:
            raise HTTPException(404)
        return Response(p.read_bytes(), media_type="video/mp4")

    @app.post("/api/lanes/{lane_id}/video")
    async def start_video(lane_id: int, body: VideoBody):
        lane = store.lane(lane_id)
        if not lane.open:
            raise HTTPException(409, f"{lane.name} is closed")
        p = VIDEOS_DIR / Path(body.name).name
        if not p.exists() or p.suffix.lower() not in VIDEO_EXT:
            raise HTTPException(404, f"Video '{body.name}' not found in videos/")
        if not (0.5 <= body.interval_sec <= 30):
            raise HTTPException(400, "interval_sec must be between 0.5 and 30")
        store.stop_video(lane)
        lane.video_name = p.name
        lane.video_task = asyncio.create_task(store.play_video(lane, p, body.interval_sec, body.loop))
        await store.push_lanes()
        return lane.to_dict(time.time())

    @app.post("/api/lanes/{lane_id}/video/stop")
    async def stop_video(lane_id: int):
        lane = store.lane(lane_id)
        store.stop_video(lane)
        await store.push_lanes()
        return lane.to_dict(time.time())

    # ---------------- LED (ESP32) ----------------
    @app.get("/api/led")
    async def led():
        now = time.time()
        return {str(l.id): l.signal(now) for l in store.lanes.values()}

    @app.get("/api/led/{lane_id}", response_class=PlainTextResponse)
    async def led_one(lane_id: int):
        return store.lane(lane_id).signal(time.time())

    # ---------------- SSE ----------------
    @app.get("/api/events")
    async def events(request: Request):
        q = store.bus.subscribe()

        async def gen():
            try:
                yield f"event: snapshot\ndata: {json.dumps({'health': await health(), 'lanes': store.snapshot(), 'model': engine.regressor.summary(), 'log': store.log[:30]})}\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        yield await asyncio.wait_for(q.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield ": ping\n\n"
            finally:
                store.bus.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description="QueueIQ Vision API")
    # PaaS builders (Nixpacks, Heroku, Railway, Coolify) hand the port in $PORT
    # and expect the process to listen on every interface, so honour that
    # convention when it is present without changing the local default.
    paas_port = os.environ.get("PORT", "").strip()
    parser.add_argument("--host",
                        default=os.environ.get("QUEUEIQ_HOST") or ("0.0.0.0" if paas_port else "127.0.0.1"),
                        help="127.0.0.1 (default) or 0.0.0.0 for the LAN; "
                             "defaults to 0.0.0.0 when $PORT is set")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("QUEUEIQ_PORT") or paas_port or 8000))
    parser.add_argument("--mode", choices=["auto", "mock"], default=os.environ.get("QUEUEIQ_MODE", "auto"))
    args = parser.parse_args()
    app = create_app(mode=args.mode)
    tier = app.state.store.engine.tier
    print(f"QueueIQ Vision API -> http://{args.host}:{args.port}  (tier={tier})")
    if app.state.store.engine.load_error:
        print(f"WARNING: models not loaded, running placeholder detections: "
              f"{app.state.store.engine.load_error}")
    if os.environ.get("QUEUEIQ_PUBLIC", "").strip().lower() in ("1", "true", "yes"):
        print("WARNING: QUEUEIQ_PUBLIC is set — the private-network guard is disabled. "
              "Only do this behind a reverse proxy that authenticates callers.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
