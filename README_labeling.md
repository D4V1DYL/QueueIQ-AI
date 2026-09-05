# Panduan CSV Labeling — Queue & Basket

## Kolom

| Kolom | Tipe | Nilai valid | Keterangan |
|---|---|---|---|
| `image_id` | text | bebas, unik per gambar/frame | Identifier sumber gambar (nama file/frame video) |
| `person_id` | text | P1, P2, P3, ... | ID orang dalam 1 gambar, urut sesuai posisi |
| `queue_position` | int | 1 = paling depan (dekat kasir) | Urutan antrian, isi manual saat labeling |
| `has_basket` | text | `yes` / `no` | Apakah orang tsb kelihatan bawa basket/troli |
| `basket_type` | text | `basket` / `cart` / `none` | Jenis wadah belanja |
| `fullness_label` | text | `empty` / `light` / `medium` / `full` / `no_basket_with_items` | Kelas kepenuhan basket (dipakai buat classifier) |
| `estimated_item_count` | int | perkiraan manual jumlah barang | Dipakai sebagai ground truth kasar saat belum ada data real |
| `predicted_checkout_time_sec` | int | hasil formula/random baseline | **Ini yang nanti diganti model asli setelah fine-tuning** |
| `actual_checkout_time_sec` | int | kosong dulu | Diisi belakangan dari data real (stopwatch/log kasir) |
| `prediction_source` | text | `random_baseline` / `heuristic` / `model_v1` dst | Penanda versi metode prediksi, penting buat tracking evaluasi |
| `notes` | text | bebas | Catatan tambahan (barang menonjol, kasus aneh, dll) |

## Formula random-tapi-masuk-akal (dipakai sekarang, sementara)

Karena belum ada data real, `predicted_checkout_time_sec` dihitung pakai formula
berbasis `estimated_item_count`, ditambah noise acak supaya tidak terkesan
kaku/linear sempurna:

```
base_time = 20 + (item_count * 7)       # detik dasar: 20s overhead + 7s/item
noise     = random(-10%, +10%) dari base_time
predicted_checkout_time_sec = base_time + noise
```

Contoh:
- 1 item  → base 27s  → hasil ~24-30s
- 4 item  → base 48s  → hasil ~43-53s
- 9 item  → base 83s  → hasil ~75-91s
- 17 item → base 139s → hasil ~125-153s

Ini sengaja dibuat **monoton naik terhadap jumlah barang** (bukan pure random),
supaya logikanya tetap masuk akal buat demo, meski angkanya belum dari data asli.

## Alur fine-tuning ke depan

1. **Sekarang (hackathon):** isi `predicted_checkout_time_sec` pakai formula di atas,
   `prediction_source = random_baseline`, `actual_checkout_time_sec` dikosongkan.
2. **Saat praktek/demo real:** setiap kali ada customer selesai checkout beneran,
   catat waktu aktualnya ke `actual_checkout_time_sec`.
3. **Setelah cukup banyak baris terisi actual time** (idealnya 30+ baris),
   baru training regresi sederhana (linear regression atau lookup table
   per rentang item count) dari `estimated_item_count` → `actual_checkout_time_sec`.
4. Ganti `prediction_source` jadi `model_v1`, `model_v2`, dst tiap kali model
   di-retrain, biar kelihatan progresnya di dashboard/demo (bagus buat judges —
   nunjukkin sistem "belajar").
