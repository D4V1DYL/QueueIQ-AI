"""
online_learning_simulation.py
Simulates the system "learning" from checkout transactions one at a time
(online learning), without retraining the whole model whenever new data arrives.

How it works (ONLINE LINEAR REGRESSION, not just an average per category):
    predicted_time = intercept + slope * item_count

1. Start from a deliberately inaccurate guess (intercept=10, slope=4),
   representing the initial heuristic formula before any real data.
2. For every new transaction compute the error (actual - predicted), then
   update the intercept & slope with stochastic gradient descent (SGD):
       intercept += lr_intercept * error
       slope     += lr_slope     * error * item_count
3. Because the data really is built from a linear relation + noise
   (see generate_synthetic_data.py: base_time = 20 + 7*items), the parameters
   converge towards (20, 7) as data accumulates -> the error shrinks.

Can be rerun whenever synthetic_checkout_data.csv is replaced by real data;
the mechanism is generic and does not depend on the synthetic numbers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LR_INTERCEPT = 0.15   # learning rate for the intercept
LR_SLOPE = 0.0015     # learning rate for the slope (smaller, it is multiplied by item_count)

INITIAL_INTERCEPT = 10.0  # initial guess (inaccurate on purpose, far from the true 20)
INITIAL_SLOPE = 4.0       # initial guess (on purpose far from the true 7)


def run_simulation(df: pd.DataFrame,
                    lr_intercept: float = LR_INTERCEPT,
                    lr_slope: float = LR_SLOPE) -> pd.DataFrame:
    intercept = INITIAL_INTERCEPT
    slope = INITIAL_SLOPE
    results = []

    for _, row in df.iterrows():
        item_count = row["estimated_item_count"]
        actual = row["actual_checkout_time_sec"]

        # predict BEFORE seeing the actual value, with the current parameters
        predicted = intercept + slope * item_count
        predicted = max(10, predicted)  # floor so it never goes negative early on

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

        # update the parameters AFTER seeing the actual value (online learning / SGD step)
        intercept += lr_intercept * error
        slope += lr_slope * error * item_count

    return pd.DataFrame(results)


def compute_rolling_accuracy(results: pd.DataFrame, window: int = 15) -> pd.Series:
    # accuracy = 100% - mean percentage error, over a rolling window
    rolling_error = results["abs_pct_error"].rolling(window=window, min_periods=5).mean()
    return 100 - rolling_error


def plot_improvement(results: pd.DataFrame, rolling_acc: pd.Series, out_path: str):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(results["transaction_id"], results["predicted_checkout_time_sec"],
                 label="Predicted", marker="o", markersize=3, linewidth=1)
    axes[0].plot(results["transaction_id"], results["actual_checkout_time_sec"],
                 label="Actual", marker="x", markersize=3, linewidth=1, alpha=0.6)
    axes[0].set_ylabel("Checkout time (sec)")
    axes[0].set_title("Predicted vs Actual Checkout Time per Transaction")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(results["transaction_id"], rolling_acc, color="green", linewidth=2)
    axes[1].axhline(90, color="gray", linestyle="--", linewidth=1, label="90% target")
    axes[1].set_ylabel("Prediction Accuracy (%)")
    axes[1].set_xlabel("Transaction #")
    axes[1].set_title("Rolling Prediction Accuracy (window=15) — the system learns over time")
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
    print(f"\nAccuracy, first 10 transactions: {first_10_acc:.1f}%")
    print(f"Accuracy, last 10 transactions : {last_10_acc:.1f}%")
    print(f"\nDetailed results -> online_learning_results.csv")
