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
| Simulasi online learning | `python online_learning_simulation.py` |

Foto dimasukkan ke `dataset/raw/{empty,light,medium,full,no_basket_with_items}/`
(nama folder = label). Panduan pengambilan foto: [`PANDUAN_FOTO.md`](PANDUAN_FOTO.md).

## Status

- ✅ Basket fullness classifier (MobileNetV2 transfer learning, 9.2 MB, 5.6 ms/gambar di GTX 1050 Ti)
- ✅ Prediksi waktu checkout — online linear regression, akurasi 74%→88% dalam 120 transaksi simulasi
- ✅ Person detection + skor antrian (`detect_queue.py`): YOLOv8n pretrained → crop area bawaan per orang → fullness → estimasi tunggu → status lane 🟢🟡🔴
- ✅ Basket detector fine-tuned (`finetune_basket_detector.py`): auto-label YOLO-World (zero-shot, tanpa anotasi manual) → fine-tune YOLOv8n — mAP50 0.887, precision 0.93; `detect_queue.py` otomatis memakainya bila `basket_detector.pt` ada
- ⬜ Queue zone + tracking (ByteTrack) · dashboard Next.js + Supabase · LED ESP32
