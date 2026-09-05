# MASTER PROMPT — AI-Powered Smart Checkout Queue System

Gunakan prompt ini secara utuh untuk brief ke AI coding assistant (Claude Code,
ChatGPT, dll), atau sebagai dokumen acuan tim hackathon.

---

## 1. Konsep Proyek

Bangun sistem **AI-Powered Smart Checkout Queue System** untuk supermarket
dengan beberapa jalur kasir (checkout lane). Tiap lane punya:
- Kamera overhead (top-down view) memantau antrian.
- Indikator lampu fisik 3 warna: 🟢 tercepat, 🟡 normal, 🔴 lambat.

Tujuan: pakai computer vision + prediksi untuk menentukan lane mana yang
diprediksi paling cepat, lalu komunikasikan lewat lampu fisik + dashboard
real-time. Skor tidak boleh cuma menghitung jumlah orang — dua customer
dengan keranjang penuh bisa lebih lambat dari lima customer dengan sedikit
barang.

## 2. Scope MVP (24-48 jam hackathon)

**Prioritas — kerjakan ini dulu:**
1. Person detection di antrian (pakai model pretrained, TIDAK perlu training ulang).
2. Cart/basket detection (fine-tune ringan dari dataset publik).
3. Basket fullness classifier: `empty` / `light` / `medium` / `full` (BUKAN deteksi per-item barang).
4. Queue tracking sederhana (ByteTrack/DeepSORT di atas hasil deteksi person).
5. Checkout Load/Queue Score per lane, dihitung dari jumlah orang + fullness rata-rata.
6. Prediksi waktu checkout: online linear regression yang belajar dari feedback
   (predicted_time = intercept + slope × jumlah_item), di-update tiap ada
   actual_checkout_time_sec baru.
7. Dashboard customer-facing (status lane + estimasi waktu) dan manager-facing
   (detail per lane + root cause kenapa lane lambat).
8. Kontrol lampu fisik (ESP32/Arduino) yang berubah warna sesuai skor real-time.

**JANGAN kerjakan ini di MVP (terlalu berat untuk waktu terbatas):**
- Object detection presisi per barang individual di keranjang (oklusi terlalu berat, butuh 300-500+ gambar dataset).
- Model deep learning kompleks (LSTM dll) untuk prediksi waktu — data terlalu sedikit, akan overfit. Pakai model linear/regresi sederhana saja.

## 3. Computer Vision Pipeline — Detail Teknis

| Komponen | Pendekatan | Dataset yang dipakai | Data custom dibutuhkan |
|---|---|---|---|
| Person detection | YOLOv8/v11 pretrained (COCO), filter class `person` | Tidak perlu | 0 gambar |
| Cart/basket detection | Fine-tune dari dataset publik (Smart Cart 4 / RPC checkout dataset di Roboflow Universe) | Ya, sebagai basis | 30-50 gambar dari sudut kamera sendiri |
| Basket fullness | Classifier (MobileNet/ResNet, transfer learning), 4 kelas | Sebagian, atau rekam sendiri | 80-150 gambar (20-40/kelas), atau heuristik CV (background subtraction) untuk versi tercepat, 0 gambar |
| Queue tracking | Algoritmik (ByteTrack/DeepSORT), bukan ML terlatih | Tidak perlu | 0 |
| Prediksi waktu checkout | Online linear regression (SGD update per transaksi) | Sintetis (lihat bagian 5) | 20-30 titik data untuk kalibrasi awal |

Kalau waktu sangat mepet, basket fullness bisa diganti heuristik computer
vision klasik (hitung area piksel non-background di bounding box keranjang)
tanpa training model sama sekali — trade-off: akurasi lebih kasar tapi 0
dataset dan 0 waktu training.

### 3.1 Basket Fullness Classifier — Implementasi

Sudah tersedia kode training + inference (belum di-training karena belum ada
foto asli — tinggal jalankan begitu dataset foto sudah dikumpulkan):

- **`train_fullness_classifier.py`** — transfer learning dari MobileNetV2
  (pretrained ImageNet), cuma melatih ulang layer classifier terakhir.
  Cocok untuk dataset kecil (80-150 gambar), ringan, jalan di CPU biasa
  (gak wajib GPU) — penting untuk demo hackathon di laptop.
  Input: folder `dataset/train/{empty,light,medium,full}/` dan
  `dataset/val/{empty,light,medium,full}/`. Output: `fullness_classifier.pt`
  + `class_names.txt`.
- **`predict.py`** — inference ke foto baru (satu file atau satu folder
  sekaligus), output kelas prediksi + confidence score per kelas.

**Cara pakai (di laptop/Google Colab, BUKAN di lingkungan chat ini karena
tidak ada PyTorch terpasang & belum ada foto training):**
```
pip install torch torchvision pillow

python train_fullness_classifier.py --data_dir ./dataset --epochs 15
python predict.py --image ./contoh_basket.jpg
```

Kalau target hackathon terlalu mepet untuk kumpulkan 80-150 foto berlabel,
fallback ke heuristik CV (background subtraction / hitung area piksel
terisi di bounding box basket) sebagai pengganti sementara — 0 dataset,
0 training, tinggal ganti lagi ke classifier ini begitu foto sudah cukup.

## 4. Skema Label & Anotasi

**Bounding box (format YOLO):**
- `person`
- `cart`, `basket`
- `person-with-cart` (opsional)

**Klasifikasi fullness (per crop basket):**
- `empty`, `light` (1-5 item), `medium` (6-15 item), `full` (16+ item)
- `no_basket_with_items` (kasus customer bawa barang di tangan tanpa keranjang — WAJIB diantisipasi, sering terjadi di kondisi nyata)

**CSV logging per transaksi (kolom):**
```
image_id, person_id, queue_position, has_basket, basket_type,
fullness_label, estimated_item_count, predicted_checkout_time_sec,
actual_checkout_time_sec, prediction_source, notes
```

## 5. Strategi Data — Tidak Ada Data Real Saat Hackathon

Karena tidak ada akses data supermarket asli, gunakan **synthetic data yang
"masuk akal secara statistik"**, bukan random murni:

- Generate ~100-120 transaksi sintetis: waktu dasar dari formula
  `20 + 7 × jumlah_item`, ditambah variasi kecepatan kasir (noise normal
  ±8-25%) dan ~12% transaksi dengan outlier delay (30-90 detik, simulasi
  kartu ditolak/masalah pembayaran).
- Jalankan online-learning simulation di atas data ini secara berurutan
  (seolah masuk satu-satu dari waktu ke waktu) untuk membuktikan mekanisme
  belajar bekerja: parameter model harus konvergen mendekati nilai asli
  formula, dan prediction accuracy harus naik seiring waktu (target dari
  ~70% ke ~85-90%+).
- Tampilkan grafik "prediction accuracy improvement" ini di dashboard
  manager sebagai bukti sistem "belajar dari feedback".
- **Transparansi ke judges:** jelaskan bahwa data ini synthetic untuk
  membuktikan mekanisme, dan begitu sistem deployed, mekanisme yang sama
  otomatis belajar dari data checkout asli tanpa perlu ubah kode.

## 6. Arsitektur Sistem

```
OVERHEAD CAMERA
  → COMPUTER VISION (Python, OpenCV + YOLO)
    → Person / Cart / Basket Fullness Detection
      → Queue Tracking (ByteTrack)
        → Prediction Engine (online linear regression)
          → Checkout Load / Wait-Time Score
            ├─→ LED CONTROLLER (ESP32/Arduino) — 🟢🟡🔴
            └─→ DASHBOARD (Next.js + Supabase Realtime)
                 ├─ Customer-facing: status lane + estimasi waktu
                 └─ Manager-facing: detail per lane, root cause, accuracy trend
```

**Stack:**
- Computer vision: Python, OpenCV, YOLOv8 (Ultralytics)
- Prediction engine: Python (numpy/pandas), online SGD regression
- Backend/realtime: Supabase (Postgres + Realtime subscriptions)
- Frontend: Next.js
- Hardware: ESP32/Arduino untuk LED, webcam/HP untuk kamera overhead saat demo
- Komunikasi: WebSocket / Supabase Realtime channel

## 7. Skema Database (Supabase) — Starting Point

```sql
-- Tabel transaksi checkout (log per customer selesai checkout)
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

-- Tabel status lane real-time
create table lane_status (
  lane_id int primary key,
  current_queue_count int,
  avg_fullness_score numeric,
  predicted_wait_sec numeric,
  status text check (status in ('green','yellow','red')),
  updated_at timestamptz default now()
);

-- Tabel parameter model (untuk online learning, persist antar restart)
create table model_parameters (
  id bigint generated always as identity primary key,
  intercept numeric,
  slope numeric,
  updated_at timestamptz default now()
);
```

## 8. Demo Flow untuk Presentasi ke Judges

1. Tunjukkan kamera live/rekaman → deteksi person + basket real-time di layar.
2. Tunjukkan basket fullness classifier bekerja pada beberapa contoh keranjang berbeda isi.
3. Tunjukkan dashboard dengan skor per lane dan lampu LED fisik berubah warna otomatis.
4. Tunjukkan grafik prediction accuracy improvement dari simulasi synthetic data — jelaskan mekanisme online learning secara transparan.
5. Tunjukkan skenario "before-after": lampu berubah dari 🟡 ke 🔴 saat satu lane tiba-tiba dapat customer dengan keranjang penuh.
6. Tutup dengan root-cause explanation di manager dashboard (misal "Line 3 lambat karena HIGH CUSTOMER LOAD, bukan cuma jumlah orang banyak").

## 9. Poin Diferensiasi (untuk pitch ke judges)

- Bukan sekadar people counter — mempertimbangkan **isi keranjang**, bukan cuma jumlah orang.
- Sistem **belajar dari feedback real** (online learning), bukan skor statis.
- Transparan soal keterbatasan data saat hackathon, tapi arsitekturnya siap pakai data real tanpa perubahan kode.
- Kombinasi computer vision + prediksi + hardware fisik (LED) + dashboard — end-to-end, bukan cuma model ML terisolasi.

## 10. File Pendukung yang Sudah Disiapkan
- `queue_checkout_labeling_template.csv` / `queue_checkout_labeling_blank.csv` — template anotasi manual.
- `generate_synthetic_data.py` — generator data checkout sintetis.
- `online_learning_simulation.py` — simulasi online learning + grafik accuracy improvement.
- `README_labeling.md` — panduan kolom & formula prediksi awal.
- `train_fullness_classifier.py` — training basket fullness classifier (transfer learning MobileNetV2). Belum dijalankan, menunggu dataset foto asli.
- `predict.py` — inference basket fullness classifier ke foto baru.

**Status implementasi saat ini:**
| Komponen | Status |
|---|---|
| Person/cart detection (YOLO pretrained) | Belum diimplementasi di repo, tinggal pakai model pretrained langsung |
| Basket fullness classifier | Kode training/inference sudah ada, belum di-training (menunggu foto) |
| Prediksi waktu checkout (online regression) | Sudah diimplementasi & diuji dengan synthetic data |
| Dashboard Next.js + Supabase | Belum diimplementasi |
| Kontrol LED (ESP32/Arduino) | Belum diimplementasi |
