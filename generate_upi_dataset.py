# generate_upi_dataset.py

import numpy as np
import pandas as pd
from pathlib import Path


def generate_upi_data(n_rows=20000, fraud_ratio=0.03, random_state=42):
    np.random.seed(random_state)

    # Some dummy UPI handles, locations, txn types
    upi_handles = ["@okicici", "@oksbi", "@okaxis", "@ybl", "@paytm", "@okhdfcbank"]
    cities = ["Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Pune", "Jaipur"]
    txn_types = ["P2P", "Merchant", "Bill", "QR_Scan"]

    data = []

    n_fraud = int(n_rows * fraud_ratio)
    n_normal = n_rows - n_fraud

    def random_upi():
        name = "".join(np.random.choice(list("abcdefghijklmnopqrstuvwxyz"), size=np.random.randint(5, 10)))
        handle = np.random.choice(upi_handles)
        return name + handle

    # ---------- NORMAL TRANSACTIONS ----------
    for _ in range(n_normal):
        sender = random_upi()
        receiver = random_upi()

        amount = np.round(np.random.exponential(scale=800), 2)
        amount = max(10, min(amount, 100000))  # clamp

        hour_of_day = np.random.randint(0, 24)
        day_of_week = np.random.randint(0, 7)
        city = np.random.choice(cities)
        txn_type = np.random.choice(txn_types)

        device_id = "dev_" + str(np.random.randint(1, 5000))
        is_new_device = np.random.choice([0, 1], p=[0.9, 0.1])

        num_txn_last_1hr = np.random.poisson(lam=1)
        num_txn_last_24hr = num_txn_last_1hr + np.random.poisson(lam=3)

        avg_amount_last_week = np.round(np.random.uniform(100, 5000), 2)
        is_new_receiver = np.random.choice([0, 1], p=[0.8, 0.2])

        label = 0  # normal

        data.append([
            sender, receiver, amount, hour_of_day, day_of_week, city, txn_type,
            device_id, is_new_device, num_txn_last_1hr, num_txn_last_24hr,
            avg_amount_last_week, is_new_receiver, label
        ])

    # ---------- FRAUD TRANSACTIONS ----------
    for _ in range(n_fraud):
        sender = random_upi()
        receiver = random_upi()

        # suspiciously high amount (or way out of pattern)
        if np.random.rand() < 0.7:
            amount = np.round(np.random.uniform(10000, 200000), 2)
        else:
            amount = np.round(np.random.uniform(5000, 10000), 2)

        # late night / odd hours
        hour_of_day = np.random.choice([0, 1, 2, 3, 23])

        day_of_week = np.random.randint(0, 7)
        city = np.random.choice(cities)
        txn_type = np.random.choice(["P2P", "QR_Scan"])  # common fraud routes

        device_id = "dev_" + str(np.random.randint(1, 8000))
        is_new_device = np.random.choice([0, 1], p=[0.3, 0.7])

        num_txn_last_1hr = np.random.randint(5, 20)
        num_txn_last_24hr = num_txn_last_1hr + np.random.randint(5, 30)

        avg_amount_last_week = np.round(np.random.uniform(100, 5000), 2)
        is_new_receiver = np.random.choice([0, 1], p=[0.2, 0.8])

        label = 1  # fraud

        data.append([
            sender, receiver, amount, hour_of_day, day_of_week, city, txn_type,
            device_id, is_new_device, num_txn_last_1hr, num_txn_last_24hr,
            avg_amount_last_week, is_new_receiver, label
        ])

    columns = [
        "sender_upi", "receiver_upi", "amount", "hour_of_day", "day_of_week",
        "location", "txn_type", "device_id", "is_new_device",
        "num_txn_last_1hr", "num_txn_last_24hr", "avg_amount_last_week",
        "is_new_receiver", "label"
    ]

    df = pd.DataFrame(data, columns=columns)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df


def main():
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    df = generate_upi_data()
    df.to_csv(out_dir / "upi_transactions_dataset.csv", index=False)

    print(df.head())
    print("Total rows:", len(df))
    print("Fraud count:", int(df["label"].sum()))
    print("✅ Saved dataset to data/upi_transactions_dataset.csv")


if __name__ == "__main__":
    main()

