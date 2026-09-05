# Panduan Pengambilan Foto — Basket Fullness Dataset (±150 foto)

Target: **±150 foto** untuk melatih basket fullness classifier
(`empty` / `light` / `medium` / `full`). Foto pakai kamera HP saja sudah cukup.

## 1. Target jumlah per kelas

Bagi rata, sisakan sedikit untuk kasus khusus:

| Kelas | Definisi | Target foto |
|---|---|---|
| `empty` | keranjang kosong | 30 |
| `light` | 1–5 item | 35 |
| `medium` | 6–15 item | 35 |
| `full` | 16+ item | 35 |
| `no_basket_with_items` | barang dijinjing tanpa keranjang | 15 (opsional tapi sangat disarankan) |

Kalau tidak sampai 150, **prioritaskan keseimbangan antar kelas** daripada
total. 25 foto per kelas yang seimbang lebih baik daripada 60 `empty` + 15 `full`.

## 2. Sudut & posisi kamera

- **Utamakan top-down / overhead** (kamera dari atas, ±45–90° dari horizontal) —
  ini menyamai posisi kamera sebenarnya di atas antrian kasir.
- Jarak ±1–2.5 meter dari keranjang, keranjang terlihat utuh di frame.
- Sebagian foto boleh agak miring (oblique) — kamera asli juga tidak selalu
  tegak lurus sempurna.

## 3. Variasi yang WAJIB ada (ini yang bikin model kuat)

Untuk tiap kelas, variasikan:

- **Isi keranjang**: jenis barang beda-beda (botol, kemasan, sayur, kotak).
  Jangan 35 foto `medium` dengan isi yang sama persis.
- **Pencahayaan**: terang, agak redup, cahaya lampu vs cahaya jendela.
- **Latar belakang**: lantai beda warna/tekstur, dekat kasir, di lorong.
- **Jenis keranjang**: keranjang jinjing + trolley kalau bisa dua-duanya.
- **Oklusi ringan**: sebagian foto ada tangan/badan orang menutupi sebagian
  keranjang — kondisi nyata memang begitu.

Hindari: foto blur parah, keranjang terpotong lebih dari setengah, atau
duplikat (burst 10 foto dari posisi sama = efektifnya cuma 1 data).

## 4. Teknis file

- Format: `.jpg` / `.png` / `.webp` — bebas.
- Nama file bebas (tidak perlu rename), tapi jangan ada nama kembar di satu folder.
- Resolusi HP standar sudah cukup (nanti di-resize ke 224×224 saat training).

## 5. Alur setelah foto terkumpul

1. Masukkan foto ke folder kelasnya:
   ```
   dataset/raw/empty/
   dataset/raw/light/
   dataset/raw/medium/
   dataset/raw/full/
   dataset/raw/no_basket_with_items/   (kalau ada)
   ```
2. Split otomatis ke train/val (80/20):
   ```
   python prepare_dataset.py
   ```
3. Training (di laptop/Colab yang ada PyTorch):
   ```
   pip install torch torchvision pillow
   python train_fullness_classifier.py --data_dir ./dataset --epochs 15
   ```
4. Tes ke foto baru:
   ```
   python predict.py --image ./foto_tes.jpg
   ```

Foto boleh masuk bertahap — `prepare_dataset.py` aman dijalankan berulang
kali; tiap run dia membuang split lama dan membagi ulang dari `raw/`.
Foto asli di `raw/` tidak pernah diubah atau dihapus.

## 6. Ambang batas kapan mulai training

- **< 20 foto/kelas**: belum layak — kumpulkan dulu.
- **20–30 foto/kelas**: sudah bisa training pertama, akurasi kasar tapi cukup buat demo awal.
- **30–40 foto/kelas** (target 150 total): sweet spot untuk transfer learning MobileNetV2.
