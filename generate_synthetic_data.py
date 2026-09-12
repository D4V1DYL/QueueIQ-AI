"""
generate_synthetic_data.py
Creates "plausible" synthetic checkout data (not pure random) for the
simulation, because there is no real supermarket data yet.

Logic:
- Base time computed from the item count (linear formula).
- Plus realistic noise: cashier-speed variation, and occasional delays
  (payment problems, price checks, etc.) appearing randomly as outliers.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_TRANSACTIONS = 120  # number of simulated checkout transactions

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
        # item count: gamma distribution (many small transactions, a few big
        # shops) -- realistic for a supermarket
        item_count = int(np.clip(rng.gamma(shape=2.0, scale=5.0), 1, 30))

        # base time: 20 s overhead + 7 s per item (normal-speed cashier)
        base_time = 20 + item_count * 7

        # cashier speed varies between transactions (multiplier ~0.85x - 1.15x)
        cashier_speed_factor = rng.normal(loc=1.0, scale=0.08)
        cashier_speed_factor = np.clip(cashier_speed_factor, 0.8, 1.25)

        actual_time = base_time * cashier_speed_factor

        # random outliers: ~12% of transactions hit an unexpected delay
        # (declined card, price check, etc.) -> extra 30-90 seconds
        if rng.random() < 0.12:
            actual_time += rng.uniform(30, 90)

        actual_time = max(15, round(actual_time))  # at least 15 seconds, a sensible floor

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
