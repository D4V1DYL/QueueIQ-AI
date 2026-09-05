"""
generate_synthetic_data.py
Membuat data checkout sintetis yang "masuk akal" (bukan random murni),
untuk simulasi karena belum ada data supermarket asli.

Logika:
- Waktu dasar dihitung dari jumlah item (formula linear).
- Ditambah noise realistis: variasi kecepatan kasir, kadang ada delay
  (masalah pembayaran, tanya harga, dll) yang muncul acak sebagai outlier.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_TRANSACTIONS = 120  # jumlah transaksi checkout yang disimulasikan

FULLNESS_BINS = [
    ("empty", 0, 0),
    ("light", 1, 5),
    ("medium", 6, 15),
    ("full", 16, 30),
]


def fullness_label(item_count: int) -> str:
    for label, lo, hi in FULLNESS_BINS:
        if lo <= item_count <= hi:
            return label
    return "full"


def generate_synthetic_checkouts(n=N_TRANSACTIONS, seed=RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    rows = []
    for i in range(1, n + 1):
        # jumlah item: distribusi gamma (lebih banyak transaksi kecil,
        # sedikit transaksi belanja besar) -- realistis untuk supermarket
        item_count = int(np.clip(rng.gamma(shape=2.0, scale=5.0), 1, 30))

        # waktu dasar: overhead 20 detik + 7 detik per item (asumsi kasir normal)
        base_time = 20 + item_count * 7

        # kecepatan kasir bervariasi antar transaksi (multiplier ~0.85x - 1.15x)
        cashier_speed_factor = rng.normal(loc=1.0, scale=0.08)
        cashier_speed_factor = np.clip(cashier_speed_factor, 0.8, 1.25)

        actual_time = base_time * cashier_speed_factor

        # outlier acak: ~12% transaksi kena delay tak terduga
        # (kartu ditolak, tanya harga, dsb) -> tambahan 30-90 detik
        if rng.random() < 0.12:
            actual_time += rng.uniform(30, 90)

        actual_time = max(15, round(actual_time))  # minimal 15 detik, sensible floor

        rows.append({
            "transaction_id": i,
            "estimated_item_count": item_count,
            "fullness_label": fullness_label(item_count),
            "actual_checkout_time_sec": int(actual_time),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_synthetic_checkouts()
    df.to_csv("synthetic_checkout_data.csv", index=False)
    print(f"Generated {len(df)} synthetic transactions -> synthetic_checkout_data.csv")
    print(df.groupby("fullness_label")["actual_checkout_time_sec"].agg(["count", "mean", "min", "max"]))
