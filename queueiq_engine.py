"""
queueiq_engine.py

Inti pipeline QueueIQ dalam bentuk library (dipakai oleh server.py, bisa juga
diimpor dari notebook/skrip lain). Menggabungkan:

    frame kamera
      -> YOLO pretrained (COCO): deteksi PERSON
      -> (opsional) basket detector hasil fine-tune
      -> basket fullness classifier  ATAU  heuristik CV bila model belum ada
      -> online linear regression: estimasi detik checkout per orang
      -> skor lane + status lampu hijau / kuning / merah

Engine memilih "tier" secara otomatis sesuai apa yang tersedia di mesin:

    full       YOLO + fullness_classifier.pt        (akurasi terbaik)
    heuristic  YOLO + heuristik CV (edge/warna)     (belum ada classifier)
    mock       tanpa torch sama sekali              (demo UI saja)

Tier selalu dilaporkan ke frontend lewat /health supaya transparan ke juri.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFilter, ImageStat

ROOT = Path(__file__).resolve().parent

# ---------------- konfigurasi ----------------
YOLO_WEIGHTS = ROOT / "yolov8n.pt"
BASKET_WEIGHTS = ROOT / "basket_detector.pt"
FULLNESS_MODEL = ROOT / "fullness_classifier.pt"
CLASS_NAMES_FILE = ROOT / "class_names.txt"
RESULTS_CSV = ROOT / "online_learning_results.csv"
SYNTHETIC_CSV = ROOT / "synthetic_checkout_data.csv"
MODEL_STATE = ROOT / "model_state.json"

PERSON_CONF = 0.35
BASKET_CONF = 0.35

# fullness -> perkiraan jumlah item (titik tengah rentang label)
FULLNESS_TO_ITEMS = {
    "empty": 0, "light": 3, "medium": 10, "full": 23,
    "no_basket_with_items": 2,
}
FULLNESS_ORDER = ["empty", "light", "medium", "full", "no_basket_with_items"]

# ambang status lane (detik total antrian) -> warna lampu.
# Selaras dengan dashboard: hijau <= 2 mnt, kuning <= 4 mnt, merah > 4 mnt.
THRESHOLD_GREEN = 120
THRESHOLD_AMBER = 240

STATUS_RGB = {"green": (40, 200, 80), "amber": (240, 200, 40), "red": (230, 60, 50)}


# =====================================================================
# Geometri
# =====================================================================
def carry_region(person_box, img_w, img_h):
    """Area bawaan: separuh bawah kotak person, dilebarkan ke samping
    (keranjang dijinjing di samping badan)."""
    x1, y1, x2, y2 = person_box
    h = y2 - y1
    pad = (x2 - x1) * 0.35
    return (
        max(0, int(x1 - pad)),
        int(y1 + h * 0.45),
        min(img_w, int(x2 + pad)),
        min(img_h, int(y2 + h * 0.15)),
    )


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def foot_point(box):
    return ((box[0] + box[2]) / 2, box[3])


def parse_zone(s: str):
    pts = []
    for tok in s.replace(";", " ").split():
        x, y = tok.split(",")
        pts.append((float(x), float(y)))
    if len(pts) < 3:
        raise ValueError("Zone butuh minimal 3 titik")
    return pts


def in_polygon(pt, poly):
    x, y = pt
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def status_for(total_sec: float) -> str:
    if total_sec <= THRESHOLD_GREEN:
        return "green"
    if total_sec <= THRESHOLD_AMBER:
        return "amber"
    return "red"


# =====================================================================
# Online linear regression (belajar dari feedback checkout)
# =====================================================================
@dataclass
class OnlineRegressor:
    """predicted_sec = intercept + slope * item_count, di-update SGD per transaksi.

    Sama persis dengan online_learning_simulation.py, tapi bisa disimpan/dimuat
    dan menyimpan riwayat supaya grafik "accuracy improvement" bisa ditampilkan
    di dashboard manager.
    """

    intercept: float = 10.0
    slope: float = 4.0
    lr_intercept: float = 0.15
    lr_slope: float = 0.0015
    history: list[dict] = field(default_factory=list)
    n_synthetic: int = 0

    def predict(self, item_count: float) -> float:
        return max(10.0, self.intercept + self.slope * item_count)

    def update(self, item_count: float, actual_sec: float, source: str = "live") -> dict:
        predicted = self.predict(item_count)
        error = actual_sec - predicted
        abs_pct = abs(error) / max(actual_sec, 1e-6) * 100
        before = (self.intercept, self.slope)
        self.intercept += self.lr_intercept * error
        self.slope += self.lr_slope * error * item_count
        rec = {
            "n": len(self.history) + 1,
            "item_count": float(item_count),
            "predicted_sec": round(predicted, 1),
            "actual_sec": round(float(actual_sec), 1),
            "abs_pct_error": round(abs_pct, 2),
            "intercept_before": round(before[0], 2),
            "slope_before": round(before[1], 3),
            "intercept": round(self.intercept, 2),
            "slope": round(self.slope, 3),
            "source": source,
            "ts": time.time(),
        }
        rec["rolling_accuracy"] = self._rolling_accuracy_with(rec)
        self.history.append(rec)
        if source == "synthetic":
            self.n_synthetic += 1
        return rec

    def _rolling_accuracy_with(self, rec: dict, window: int = 15) -> float:
        errs = [h["abs_pct_error"] for h in self.history[-(window - 1):]] + [rec["abs_pct_error"]]
        return round(max(0.0, 100 - sum(errs) / len(errs)), 2)

    def summary(self) -> dict:
        h = self.history
        first = h[:10]
        last = h[-10:]
        acc = lambda rows: round(100 - sum(r["abs_pct_error"] for r in rows) / len(rows), 1) if rows else None
        return {
            "intercept": round(self.intercept, 3),
            "slope": round(self.slope, 4),
            "formula": f"{self.intercept:.1f} + {self.slope:.2f} x item",
            "n_updates": len(h),
            "n_synthetic": self.n_synthetic,
            "n_live": len(h) - self.n_synthetic,
            "accuracy_first10": acc(first),
            "accuracy_last10": acc(last),
            "history": [
                {k: r[k] for k in ("n", "item_count", "predicted_sec", "actual_sec",
                                    "abs_pct_error", "rolling_accuracy", "source",
                                    "intercept", "slope")}
                for r in h
            ],
        }

    # --- persistence ---
    def save(self, path: Path = MODEL_STATE):
        path.write_text(json.dumps({
            "intercept": self.intercept, "slope": self.slope,
            "lr_intercept": self.lr_intercept, "lr_slope": self.lr_slope,
            "n_synthetic": self.n_synthetic, "history": self.history,
        }, indent=1))

    @classmethod
    def load(cls, path: Path = MODEL_STATE) -> Optional["OnlineRegressor"]:
        if not path.exists():
            return None
        d = json.loads(path.read_text())
        return cls(intercept=d["intercept"], slope=d["slope"],
                   lr_intercept=d.get("lr_intercept", 0.15),
                   lr_slope=d.get("lr_slope", 0.0015),
                   history=d.get("history", []),
                   n_synthetic=d.get("n_synthetic", 0))

    def warm_start(self, csv_path: Path = SYNTHETIC_CSV) -> int:
        """Putar ulang data sintetis (transparan: ditandai source='synthetic')
        supaya model tidak mulai dari nol saat demo dan grafik langsung
        menunjukkan kurva belajar."""
        if not csv_path.exists():
            return 0
        n = 0
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                self.update(float(row["estimated_item_count"]),
                            float(row["actual_checkout_time_sec"]), source="synthetic")
                n += 1
        return n


# =====================================================================
# Fullness: classifier (jika ada) atau heuristik CV
# =====================================================================
class HeuristicFullness:
    """Fallback tanpa training (MASTER_PROMPT §3): kepadatan tepi + variasi
    warna di area bawaan -> empty/light/medium/full. Kasar tapi 0 dataset."""

    name = "cv-heuristic"

    def predict(self, crop: Image.Image):
        g = crop.convert("L").resize((96, 96))
        edges = g.filter(ImageFilter.FIND_EDGES)
        edge_density = ImageStat.Stat(edges).mean[0] / 255.0        # 0..1
        hsv = crop.convert("HSV").resize((96, 96))
        sat = ImageStat.Stat(hsv.getchannel("S")).mean[0] / 255.0   # 0..1
        score = 0.65 * edge_density * 4 + 0.35 * sat                 # ~0..1.5
        if score < 0.25:
            label = "empty"
        elif score < 0.45:
            label = "light"
        elif score < 0.7:
            label = "medium"
        else:
            label = "full"
        conf = 0.55 + min(0.3, abs(score - 0.45))  # kepercayaan sengaja rendah
        return label, round(conf, 2), {"score": round(score, 3),
                                        "edge_density": round(edge_density, 3),
                                        "saturation": round(sat, 3)}


class TorchFullness:
    name = "mobilenetv2-classifier"

    def __init__(self, device):
        import torch
        import torch.nn as nn
        from torchvision import models, transforms

        self.torch = torch
        self.classes = CLASS_NAMES_FILE.read_text().split()
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, len(self.classes))
        model.load_state_dict(torch.load(FULLNESS_MODEL, map_location=device))
        model.to(device).eval()
        self.model = model
        self.device = device
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(self, crop: Image.Image):
        import torch.nn.functional as F
        x = self.tf(crop).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            probs = F.softmax(self.model(x), 1)[0].cpu()
        idx = int(probs.argmax())
        return self.classes[idx], round(float(probs[idx]), 3), {
            c: round(float(p), 3) for c, p in zip(self.classes, probs)
        }


# =====================================================================
# Engine
# =====================================================================
@dataclass
class Detection:
    index: int
    person_box: Optional[tuple]
    crop_box: tuple
    source: str            # "basket" | "area-bawaan" | "mock"
    fullness: str
    confidence: float
    est_items: int
    est_sec: float
    detail: dict = field(default_factory=dict)


class QueueEngine:
    def __init__(self, mode: str = "auto", regressor: Optional[OnlineRegressor] = None):
        self.mode = mode
        self.tier = "mock"
        self.device = "cpu"
        self.yolo = None
        self.basket_yolo = None
        self.fullness = None
        self.load_error: Optional[str] = None
        self.regressor = regressor or OnlineRegressor()
        self.timings: list[float] = []

    # ---------- loading ----------
    def load(self):
        if self.mode == "mock":
            self.tier = "mock"
            return self
        try:
            import torch
            from ultralytics import YOLO
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.yolo = YOLO(str(YOLO_WEIGHTS))
            if BASKET_WEIGHTS.exists():
                self.basket_yolo = YOLO(str(BASKET_WEIGHTS))
            if FULLNESS_MODEL.exists() and CLASS_NAMES_FILE.exists():
                self.fullness = TorchFullness(self.device)
                self.tier = "full"
            else:
                self.fullness = HeuristicFullness()
                self.tier = "heuristic"
        except Exception as e:  # torch/ultralytics tidak ada -> mock
            self.load_error = f"{type(e).__name__}: {e}"
            self.tier = "mock"
            self.yolo = None
            self.fullness = None
        return self

    def info(self) -> dict:
        avg = sum(self.timings[-20:]) / len(self.timings[-20:]) if self.timings else None
        return {
            "tier": self.tier,
            "device": self.device,
            "models": {
                "person_detector": "yolov8n (COCO pretrained)" if self.yolo else None,
                "basket_detector": "yolov8n fine-tuned" if self.basket_yolo else None,
                "fullness": getattr(self.fullness, "name", None),
            },
            "thresholds": {"green_max_sec": THRESHOLD_GREEN, "amber_max_sec": THRESHOLD_AMBER},
            "avg_inference_ms": round(avg * 1000, 1) if avg else None,
            "load_error": self.load_error,
        }

    # ---------- inference ----------
    def analyze(self, img: Image.Image, zone: Optional[list] = None,
                annotate: bool = True) -> tuple[dict, Optional[bytes]]:
        t0 = time.perf_counter()
        img = img.convert("RGB")
        w, h = img.size
        if self.tier == "mock":
            dets, n_all = self._mock_detections(img)
        else:
            dets, n_all = self._real_detections(img, zone)

        total = sum(d.est_sec for d in dets)
        status = status_for(total)
        elapsed = time.perf_counter() - t0
        self.timings.append(elapsed)

        result = {
            "tier": self.tier,
            "n_person": len(dets),
            "n_detected_total": n_all,
            "total_sec": round(total, 1),
            "status": status,
            "inference_ms": round(elapsed * 1000, 1),
            "image_size": [w, h],
            "detections": [self._det_to_dict(d) for d in dets],
        }
        jpeg = self._annotate(img, dets, status, total, zone) if annotate else None
        return result, jpeg

    def _real_detections(self, img, zone):
        w, h = img.size
        det = self.yolo.predict(img, conf=PERSON_CONF, classes=[0], verbose=False)[0]
        persons = [tuple(map(int, b.xyxy[0].tolist())) for b in det.boxes]
        n_all = len(persons)
        if zone:
            persons = [p for p in persons if in_polygon(foot_point(p), zone)]

        basket_of: dict[int, tuple] = {}
        orphan = []
        if self.basket_yolo is not None:
            bdet = self.basket_yolo.predict(img, conf=BASKET_CONF, verbose=False)[0]
            for b in bdet.boxes:
                bbox = tuple(map(int, b.xyxy[0].tolist()))
                bc = center(bbox)
                if zone and not in_polygon(bc, zone):
                    continue
                if not persons:
                    orphan.append(bbox)
                    continue
                nearest = min(range(len(persons)), key=lambda i: (
                    (center(persons[i])[0] - bc[0]) ** 2 + (center(persons[i])[1] - bc[1]) ** 2))
                if nearest not in basket_of or float(b.conf) > basket_of[nearest][1]:
                    basket_of[nearest] = (bbox, float(b.conf))

        subjects = [(box, basket_of[i][0] if i in basket_of else None) for i, box in enumerate(persons)]
        subjects += [(None, bbox) for bbox in orphan]

        dets = []
        for i, (pbox, bbox) in enumerate(subjects):
            if bbox is not None:
                crop_box, src = bbox, "basket"
            else:
                crop_box, src = carry_region(pbox, w, h), "area-bawaan"
            crop = img.crop(crop_box)
            label, conf, detail = self.fullness.predict(crop)
            items = FULLNESS_TO_ITEMS.get(label, 3)
            dets.append(Detection(i, pbox, crop_box, src, label, conf, items,
                                  round(self.regressor.predict(items), 1), detail))
        return dets, n_all

    def _mock_detections(self, img):
        """Deteksi deterministik dari hash gambar — untuk demo UI tanpa torch."""
        w, h = img.size
        seed = int(hashlib.md5(img.tobytes()[:20000]).hexdigest(), 16)
        rng = random.Random(seed)
        n = rng.choice([1, 2, 2, 3, 3, 4])
        dets = []
        for i in range(n):
            bw, bh = int(w * rng.uniform(0.12, 0.2)), int(h * rng.uniform(0.4, 0.6))
            x1 = int(w * (0.1 + 0.8 * i / max(n, 1)) + rng.uniform(-w * 0.03, w * 0.03))
            y1 = int(h * rng.uniform(0.2, 0.4))
            pbox = (max(0, x1), y1, min(w, x1 + bw), min(h, y1 + bh))
            label = rng.choices(["empty", "light", "medium", "full"], [1, 4, 4, 2])[0]
            items = FULLNESS_TO_ITEMS[label]
            dets.append(Detection(i, pbox, carry_region(pbox, w, h), "mock", label,
                                  round(rng.uniform(0.6, 0.95), 2), items,
                                  round(self.regressor.predict(items), 1), {}))
        return dets, n

    @staticmethod
    def _det_to_dict(d: Detection) -> dict:
        out = asdict(d)
        out["person_box"] = list(d.person_box) if d.person_box else None
        out["crop_box"] = list(d.crop_box)
        return out

    def _annotate(self, img, dets, status, total, zone) -> bytes:
        vis = img.copy()
        d = ImageDraw.Draw(vis)
        w, _ = vis.size
        if zone:
            d.polygon([tuple(p) for p in zone], outline=(255, 80, 255), width=3)
        for i, r in enumerate(dets, 1):
            if r.person_box:
                d.rectangle(r.person_box, outline=(0, 180, 255), width=3)
                tx, ty = r.person_box[0] + 4, r.person_box[1] + 4
            else:
                tx, ty = r.crop_box[0] + 4, r.crop_box[1] + 4
            d.rectangle(r.crop_box, outline=(255, 255, 0), width=2)
            d.text((tx, ty), f"#{i} {r.fullness[:8]} {r.est_sec:.0f}s", fill=(0, 180, 255))
        d.rectangle((0, 0, w, 26), fill=STATUS_RGB[status])
        d.text((8, 6), f"LANE {status.upper()} | {len(dets)} orang | ~{total:.0f} dtk"
                       f" | {self.tier}", fill=(0, 0, 0))
        buf = io.BytesIO()
        vis.save(buf, format="JPEG", quality=85)
        return buf.getvalue()


def regressor_from_disk() -> OnlineRegressor:
    """Urutan prioritas: model_state.json (hasil belajar live) -> warm start
    dari synthetic CSV -> parameter awal formula."""
    reg = OnlineRegressor.load()
    if reg is not None:
        return reg
    reg = OnlineRegressor()
    reg.warm_start()
    return reg
