"""
detect_queue.py

Pipeline deteksi antrian kasir (MASTER_PROMPT §2 poin 1-5):

    frame kamera
      -> YOLO pretrained (COCO): deteksi PERSON        [tanpa training ulang]
      -> crop area bawaan tiap person (badan bagian bawah, tempat
         keranjang/barang biasanya dibawa)
      -> basket fullness classifier (fullness_classifier.pt)
      -> estimasi waktu checkout per orang (intercept + slope x item)
      -> skor lane + status lampu: hijau / kuning / merah

CARA PAKAI:
    python detect_queue.py --image examples/antrian_cctv.jpg
    python detect_queue.py --image dataset/unsorted/          (semua foto folder)
    python detect_queue.py --video rekaman.mp4                (proses per detik)
    python detect_queue.py --image foto.jpg --no-annotate     (tanpa simpan gambar)

Output: ringkasan antrian di terminal + gambar beranotasi *_annotated.jpg.

Catatan MVP:
- COCO tidak punya kelas "shopping basket", jadi keranjang TIDAK dideteksi
  langsung; fullness dinilai dari crop area bawaan tiap person. Fine-tune
  basket detector (30-50 foto, lihat MASTER_PROMPT §3) adalah upgrade
  berikutnya - tinggal ganti sumber kotak crop, sisa pipeline tidak berubah.
- Estimasi waktu memakai parameter hasil online learning terakhir
  (online_learning_results.csv). Saat sistem live, parameter ini terus
  ter-update tiap transaksi selesai.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms
from ultralytics import YOLO

# ---------------- konfigurasi ----------------
YOLO_WEIGHTS = "yolov8n.pt"          # nano: 6MB, cukup untuk MVP
BASKET_WEIGHTS = "basket_detector.pt"  # hasil finetune_basket_detector.py (opsional)
PERSON_CONF = 0.35                    # ambang confidence deteksi person
BASKET_CONF = 0.35                    # ambang confidence deteksi keranjang
FULLNESS_MODEL = "fullness_classifier.pt"
CLASS_NAMES_FILE = "class_names.txt"
RESULTS_CSV = "online_learning_results.csv"

# fullness -> perkiraan jumlah item (titik tengah rentang label)
FULLNESS_TO_ITEMS = {
    "empty": 0, "light": 3, "medium": 10, "full": 23,
    "no_basket_with_items": 2,
}

# ambang status lane (detik total antrian) -> warna lampu
THRESHOLD_GREEN = 120     # < 2 menit
THRESHOLD_YELLOW = 300    # < 5 menit; di atasnya merah

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def load_regression_params():
    """Ambil intercept/slope terakhir dari hasil online learning."""
    import csv
    path = Path(RESULTS_CSV)
    if not path.exists():
        return 20.0, 7.0  # fallback: nilai formula sintetis
    with open(path) as f:
        last = list(csv.DictReader(f))[-1]
    return float(last["current_intercept"]), float(last["current_slope"])


def load_fullness_model(device):
    classes = Path(CLASS_NAMES_FILE).read_text().split()
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(torch.load(FULLNESS_MODEL, map_location=device))
    model.to(device).eval()
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, classes, tf


def carry_region(person_box, img_w, img_h):
    """Area bawaan: separuh bawah kotak person, dilebarkan sedikit ke samping
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


def status_lane(total_sec):
    if total_sec < THRESHOLD_GREEN:
        return "green", "HIJAU"
    if total_sec < THRESHOLD_YELLOW:
        return "yellow", "KUNING"
    return "red", "MERAH"


STATUS_RGB = {"green": (40, 200, 80), "yellow": (240, 200, 40), "red": (230, 60, 50)}


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def analyze_image(img_path, yolo, fmodel, fclasses, ftf, device,
                  intercept, slope, annotate=True, basket_yolo=None):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    det = yolo.predict(img, conf=PERSON_CONF, classes=[0], verbose=False)[0]
    persons = [tuple(map(int, b.xyxy[0].tolist())) for b in det.boxes]

    # deteksi keranjang (kalau detector hasil fine-tune tersedia),
    # lalu pasangkan tiap keranjang ke person terdekat
    basket_of = {}
    orphan_baskets = []   # keranjang terdeteksi tapi tak ada person di frame
                          # (orang tertutup rak / di luar frame) -> tetap dihitung
    if basket_yolo is not None:
        bdet = basket_yolo.predict(img, conf=BASKET_CONF, verbose=False)[0]
        for b in bdet.boxes:
            bbox = tuple(map(int, b.xyxy[0].tolist()))
            bc = center(bbox)
            if not persons:
                orphan_baskets.append(bbox)
                continue
            nearest = min(range(len(persons)), key=lambda i: (
                (center(persons[i])[0] - bc[0]) ** 2 +
                (center(persons[i])[1] - bc[1]) ** 2))
            # simpan keranjang ber-confidence tertinggi per person
            if nearest not in basket_of or float(b.conf) > basket_of[nearest][1]:
                basket_of[nearest] = (bbox, float(b.conf))

    rows = []
    subjects = [(box, basket_of[pi][0] if pi in basket_of else None)
                for pi, box in enumerate(persons)]
    subjects += [(bbox, bbox) for bbox in orphan_baskets]

    for box, basket_box in subjects:
        if basket_box is not None:
            crop_box, src = basket_box, "basket"
        else:
            crop_box, src = carry_region(box, w, h), "area-bawaan"
        crop = img.crop(crop_box)
        x = ftf(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(fmodel(x), 1)[0].cpu()
        idx = int(probs.argmax())
        label = fclasses[idx]
        items = FULLNESS_TO_ITEMS.get(label, 3)
        est = intercept + slope * items
        rows.append({
            "person_box": box, "crop_box": crop_box, "source": src,
            "fullness": label, "conf": float(probs[idx]),
            "est_items": items, "est_sec": est,
        })

    total = sum(r["est_sec"] for r in rows)
    color, color_id = status_lane(total)

    print(f"\n{Path(img_path).name}")
    print(f"  antrian : {len(rows)} orang")
    for i, r in enumerate(rows, 1):
        print(f"    #{i} {r['fullness']:<22} ({r['conf']:.0%}) "
              f"[{r['source']}]  ~{r['est_items']} item -> {r['est_sec']:.0f} dtk")
    print(f"  estimasi total tunggu : {total:.0f} dtk ({total/60:.1f} mnt)")
    print(f"  status lane           : {color_id} [{color}]")

    if annotate and rows:
        vis = img.copy()
        d = ImageDraw.Draw(vis)
        for i, r in enumerate(rows, 1):
            d.rectangle(r["person_box"], outline=(0, 180, 255), width=3)
            d.rectangle(r["crop_box"], outline=(255, 255, 0), width=2)
            d.text((r["person_box"][0] + 4, r["person_box"][1] + 4),
                   f"#{i} {r['fullness'][:8]} {r['est_sec']:.0f}s",
                   fill=(0, 180, 255))
        # banner status lane
        d.rectangle((0, 0, w, 26), fill=STATUS_RGB[color])
        d.text((8, 6), f"LANE {color_id} | {len(rows)} orang | "
                       f"~{total:.0f} dtk", fill=(0, 0, 0))
        out = Path(img_path).with_name(Path(img_path).stem + "_annotated.jpg")
        vis.save(out, quality=90)
        print(f"  anotasi -> {out}")

    return {"n_person": len(rows), "total_sec": total, "status": color, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=str, help="File gambar atau folder")
    src.add_argument("--video", type=str, help="File video (dianalisis 1 frame/detik)")
    parser.add_argument("--no-annotate", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = YOLO(YOLO_WEIGHTS)
    basket_yolo = YOLO(BASKET_WEIGHTS) if Path(BASKET_WEIGHTS).exists() else None
    fmodel, fclasses, ftf = load_fullness_model(device)
    intercept, slope = load_regression_params()
    print(f"device={device} | model waktu: {intercept:.1f} + {slope:.2f} x item | "
          f"basket detector: {'ON (fine-tuned)' if basket_yolo else 'OFF (pakai area bawaan)'}")

    if args.image:
        p = Path(args.image)
        paths = (sorted(q for q in p.iterdir() if q.suffix.lower() in IMG_EXT)
                 if p.is_dir() else [p])
        for path in paths:
            analyze_image(path, yolo, fmodel, fclasses, ftf, device,
                          intercept, slope, annotate=not args.no_annotate,
                          basket_yolo=basket_yolo)
    else:
        # video: ekstrak 1 fps ke folder sementara lalu proses per frame
        import subprocess, tempfile
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error",
                            "-i", args.video, "-vf", "fps=1", "-q:v", "2",
                            f"{td}/frame_%04d.jpg"], check=True)
            for path in sorted(Path(td).glob("frame_*.jpg")):
                analyze_image(path, yolo, fmodel, fclasses, ftf, device,
                              intercept, slope, annotate=False,
                              basket_yolo=basket_yolo)


if __name__ == "__main__":
    main()
