"""
online_learning_simulation.py
Simulasi sistem "belajar" dari transaksi checkout satu per satu (online learning),
tanpa perlu retrain ulang seluruh model tiap kali ada data baru.

Cara kerja (ONLINE LINEAR REGRESSION, bukan sekadar rata-rata per kategori):
    predicted_time = intercept + slope * jumlah_item

1. Mulai dari tebakan awal yang SENGAJA belum akurat (intercept=10, slope=4),
   merepresentasikan formula heuristik awal sebelum ada data real.
2. Tiap transaksi baru datang, hitung error (actual - predicted), lalu update
   intercept & slope pakai stochastic gradient descent (SGD):
       intercept += lr_intercept * error
       slope     += lr_slope     * error * item_count
3. Karena data sebenarnya memang dibangun dari relasi linear + noise
   (lihat generate_synthetic_data.py: base_time = 20 + 7*item), parameter
   akan konvergen mendekati (20, 7) seiring bertambahnya data -> error mengecil.

Bisa dijalankan lagi tiap kali synthetic_checkout_data.csv diganti data real;
mekanismenya generik, tidak bergantung pada angka spesifik di synthetic data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LR_INTERCEPT = 0.15   # learning rate untuk intercept
LR_SLOPE = 0.0015     # learning rate untuk slope (lebih kecil karena dikali item_count)

INITIAL_INTERCEPT = 10.0  # tebakan awal (kurang akurat, sengaja jauh dari nilai asli 20)
INITIAL_SLOPE = 4.0       # tebakan awal (sengaja jauh dari nilai asli 7)


def run_simulation(df: pd.DataFrame,
                    lr_intercept: float = LR_INTERCEPT,
                    lr_slope: float = LR_SLOPE) -> pd.DataFrame:
    intercept = INITIAL_INTERCEPT
    slope = INITIAL_SLOPE
    results = []

    for _, row in df.iterrows():
        item_count = row["estimated_item_count"]
        actual = row["actual_checkout_time_sec"]

        # prediksi SEBELUM lihat actual, pakai parameter saat ini
        predicted = intercept + slope * item_count
        predicted = max(10, predicted)  # floor biar gak negatif di awal

        error = actual - predicted
        abs_pct_error = abs(error) / actual * 100

        results.append({
            "transaction_id": row["transaction_id"],
            "fullness_label": row["fullness_label"],
            "estimated_item_count": item_count,
            "predicted_checkout_time_sec": round(predicted, 1),
            "actual_checkout_time_sec": actual,
            "abs_pct_error": round(abs_pct_error, 2),
            "current_intercept": round(intercept, 2),
            "current_slope": round(slope, 3),
        })

        # update parameter SETELAH lihat actual (online learning / SGD step)
        intercept += lr_intercept * error
        slope += lr_slope * error * item_count

    return pd.DataFrame(results)


def compute_rolling_accuracy(results: pd.DataFrame, window: int = 15) -> pd.Series:
    # accuracy = 100% - rata-rata persentase error, dihitung per rolling window
    rolling_error = results["abs_pct_error"].rolling(window=window, min_periods=5).mean()
    return 100 - rolling_error


def plot_improvement(results: pd.DataFrame, rolling_acc: pd.Series, out_path: str):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(results["transaction_id"], results["predicted_checkout_time_sec"],
                 label="Predicted", marker="o", markersize=3, linewidth=1)
    axes[0].plot(results["transaction_id"], results["actual_checkout_time_sec"],
                 label="Actual", marker="x", markersize=3, linewidth=1, alpha=0.6)
    axes[0].set_ylabel("Checkout time (sec)")
    axes[0].set_title("Predicted vs Actual Checkout Time per Transaksi")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(results["transaction_id"], rolling_acc, color="green", linewidth=2)
    axes[1].axhline(90, color="gray", linestyle="--", linewidth=1, label="90% target")
    axes[1].set_ylabel("Prediction Accuracy (%)")
    axes[1].set_xlabel("Transaction #")
    axes[1].set_title(f"Rolling Prediction Accuracy (window=15) — Sistem 'Belajar' Seiring Waktu")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved -> {out_path}")


if __name__ == "__main__":
    df = pd.read_csv("synthetic_checkout_data.csv")
    results = run_simulation(df)
    rolling_acc = compute_rolling_accuracy(results)
    results["rolling_accuracy_pct"] = rolling_acc.round(2)

    results.to_csv("online_learning_results.csv", index=False)
    plot_improvement(results, rolling_acc, "prediction_accuracy_improvement.png")

    first_10_acc = (100 - results["abs_pct_error"].iloc[:10].mean())
    last_10_acc = (100 - results["abs_pct_error"].iloc[-10:].mean())
    print(f"\nAccuracy 10 transaksi pertama : {first_10_acc:.1f}%")
    print(f"Accuracy 10 transaksi terakhir: {last_10_acc:.1f}%")
    print(f"\nHasil detail -> online_learning_results.csv")
