# QueueIQ — AI-Powered Smart Checkout Queue System

Sistem antrian kasir pintar: kamera overhead + computer vision menilai **isi
keranjang** (bukan cuma jumlah orang) untuk memprediksi lane tercepat, lalu
mengkomunikasikannya lewat lampu 🟢🟡🔴 dan dashboard real-time.

**Mulai dari sini:** [`QueueIQ_Pipeline.ipynb`](QueueIQ_Pipeline.ipynb) —
notebook lengkap (dataset → training → evaluasi → prediksi waktu checkout)
dengan penjelasan tiap tahap. Konsep & arsitektur lengkap:
[`MASTER_PROMPT.md`](MASTER_PROMPT.md).

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

> **cu126 wajib** untuk GPU Pascal (GTX 10xx) — build PyTorch cu128+ sudah
> drop dukungan Pascal. GPU tidak wajib: inference ~32 fps di CPU.

## Alur kerja

| Langkah | Perintah |
|---|---|
| Kumpulkan foto (video → frame) | `python extract_frames.py --video vid.mp4 --kelas full` |
| Suplemen dari internet (opsional) | `python scrape_images.py` / `python scrape_videos.py --kelas _unsorted` |
| Split train/val | `python prepare_dataset.py` |
| Training (±1 menit di GPU) | `python train_fullness_classifier.py --data_dir ./dataset --epochs 15` |
| Prediksi | `python predict.py --image foto.jpg` |
| **Analisis antrian lengkap** (YOLO + classifier + estimasi tunggu) | `python detect_queue.py --image examples/antrian_cctv.jpg` |
| — dengan zona antrian (hanya hitung orang di polygon) | `python detect_queue.py --image cctv.jpg --zone "130,40 330,40 330,330 130,330"` |
| Simulasi online learning | `python online_learning_simulation.py` |
| **Server API untuk dashboard** (`QueueIQ-FE` halaman `/live`) | `pip install -r requirements-server.txt` lalu `python server.py` |
| Footage CCTV untuk kamera virtual di dashboard | `pip install yt-dlp` lalu `python fetch_demo_video.py` |

## Server API (sambungan ke frontend)

`server.py` membungkus pipeline (`queueiq_engine.py`) jadi HTTP + Server-Sent
Events di `http://127.0.0.1:8000`, dipakai halaman **/live** di QueueIQ-FE.

```bash
python server.py                  # otomatis pilih tier sesuai model yang ada
python server.py --host 0.0.0.0   # demo dari HP / laptop lain di WiFi yang sama
python server.py --mode mock      # tanpa torch: hanya untuk demo UI
```

| Tier (dilaporkan di `/health` & header dashboard) | Kondisi |
|---|---|
| `full` | `yolov8n.pt` + `fullness_classifier.pt` + `class_names.txt` ada (+ `basket_detector.pt` bila ada) — **semuanya sudah ikut di repo**, tinggal clone |
| `heuristic` | YOLO ada, classifier belum — fullness dari heuristik tepi/warna (MASTER_PROMPT §3) |
| `mock` | torch/ultralytics tidak terpasang — deteksi placeholder deterministik |

Endpoint: `GET /health` · `GET /api/lanes` · `POST /api/lanes/{id}/analyze`
(multipart `file` atau form `example=antrian_cctv.jpg`) · `GET /api/lanes/{id}/frame.jpg`
· `POST /api/lanes/{id}/complete` `{"actual_sec": 87}` (feedback → SGD step, tersimpan di
`model_state.json`) · `POST /api/lanes/{id}/open` · `GET /api/model` (parameter + riwayat
akurasi) · `POST /api/model/reset` · `POST /api/demo/scenario` `{"name": "seed"|"surge"|"cctv"|"clear"}`
· `GET /api/led/{id}` (teks `green|amber|red|closed` untuk ESP32) · `GET /api/events` (SSE).

**Kamera virtual (demo tanpa webcam/supermarket):** `python fetch_demo_video.py`
mengunduh footage CCTV kasir dari YouTube ke `videos/` (2 video sudah ikut di repo). Di
dashboard `/live` video muncul di "Sample footage · play as virtual camera";
server membaca 1 frame tiap 2 detik lewat pipeline yang sama seperti webcam
(`POST /api/lanes/{id}/video`). Taruh file `.mp4` lain di `videos/` agar ikut muncul.

**Pengaman feedback:** tombol *Done* memakai waktu terukur sejak shopper depan
mulai dilayani; kalau < 15 detik (operator menekan saat demo) transaksi dianggap
bukan checkout nyata dan model TIDAK di-update — pakai *Teach the model* dengan
angka detik eksplisit.

Foto contoh untuk galeri di dashboard diambil dari folder `examples/` (semua `.jpg/.png/.webp` selain `*_annotated`) — taruh foto antrian troli/keranjang sendiri di sana agar muncul sebagai contoh yang bisa diklik.

Server hanya melayani klien loopback / IP privat LAN; persempit lagi dengan
`QUEUEIQ_ALLOWED_IPS=192.168.1.20,192.168.1.55`. Ambang lampu selaras dengan
dashboard: hijau ≤ 120 dtk, kuning ≤ 240 dtk, merah di atasnya.

Foto dimasukkan ke `dataset/raw/{empty,light,medium,full,no_basket_with_items}/`
(nama folder = label). Panduan pengambilan foto: [`PANDUAN_FOTO.md`](PANDUAN_FOTO.md).

## Status

- ✅ Basket fullness classifier (MobileNetV2 transfer learning, 9.2 MB, 5.6 ms/gambar di GTX 1050 Ti). Latih ulang dari nol dengan data scrape (`scrape_images.py` → kurasi → `prepare_dataset.py` → `train_fullness_classifier.py --epochs 25 --lr 4e-4 --unfreeze_last 4`): 225 foto, val 5 kelas 56%, empty-vs-ada-isi 93%, 3 tingkat (kosong / sedikit / sedang-penuh) 80%, meleset ≤1 tingkat 98% — label foto stok memang bising, foto asli dari kamera sendiri akan jauh lebih baik
- ✅ Prediksi waktu checkout — online linear regression, akurasi 74%→88% dalam 120 transaksi simulasi
- ✅ Person detection + skor antrian (`detect_queue.py`): YOLOv8n pretrained → crop area bawaan per orang → fullness → estimasi tunggu → status lane 🟢🟡🔴
- ✅ Basket detector fine-tuned (`finetune_basket_detector.py`): auto-label YOLO-World (zero-shot, tanpa anotasi manual) → fine-tune YOLOv8n — mAP50 0.887, precision 0.93 pada dataset asli; latih ulang dari data scrape (116 foto, 30 epoch CPU ±12 menit) memberi mAP50 0.82, precision 0.82; `detect_queue.py` dan `server.py` otomatis memakainya bila `basket_detector.pt` ada
- ✅ Queue zone (`--zone`): hanya orang/keranjang di polygon area antrian yang dihitung — kasir & pengunjung lewat tersaring
- ✅ Server API (`server.py` + `queueiq_engine.py`): FastAPI + SSE, feedback checkout → online learning live, endpoint LED untuk ESP32, tersambung ke dashboard `/live` di QueueIQ-FE
- ⬜ Tracking antar-frame (ByteTrack) · Supabase persist · firmware LED ESP32
