# app.py

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from datetime import datetime
import sqlite3
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "fraud_logs.db"

# Load model and feature info
model = joblib.load(MODELS_DIR / "upi_fraud_model.pkl")
feature_info = joblib.load(MODELS_DIR / "feature_info.pkl")


# ---------- DB HELPERS ----------

def init_db():
    """Create SQLite table if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            amount REAL,
            hour_of_day INTEGER,
            day_of_week INTEGER,
            location TEXT,
            txn_type TEXT,
            is_new_device INTEGER,
            num_txn_last_1hr INTEGER,
            num_txn_last_24hr INTEGER,
            avg_amount_last_week REAL,
            is_new_receiver INTEGER,
            risk_score INTEGER,
            decision TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def log_transaction(row_dict):
    """Insert a transaction row into the DB."""
    conn = sqlite3.connect(DB_PATH)
    cols = ",".join(row_dict.keys())
    placeholders = ",".join(["?"] * len(row_dict))
    values = list(row_dict.values())
    conn.execute(f"INSERT INTO transactions ({cols}) VALUES ({placeholders})", values)
    conn.commit()
    conn.close()


def get_dataframe(query, params=None):
    """Read a SQL query into a pandas DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, conn, params=params or [])
    conn.close()
    return df


# ---------- MODEL INPUT / SCORING ----------

def build_input_df(form_data):
    """Convert form or JSON input into a one-row DataFrame."""
    amount = float(form_data.get("amount", 0) or 0)
    hour_of_day = int(form_data.get("hour_of_day", 12) or 12)
    day_of_week = int(form_data.get("day_of_week", 0) or 0)
    location = form_data.get("location", "Delhi") or "Delhi"
    txn_type = form_data.get("txn_type", "P2P") or "P2P"

    is_new_device = 1 if str(form_data.get("is_new_device", "0")) == "1" else 0
    is_new_receiver = 1 if str(form_data.get("is_new_receiver", "0")) == "1" else 0

    num_txn_last_1hr = int(form_data.get("num_txn_last_1hr", 0) or 0)
    num_txn_last_24hr = int(form_data.get("num_txn_last_24hr", 0) or 0)
    avg_amount_last_week = float(form_data.get("avg_amount_last_week", amount) or amount)

    data = {
        "amount": [amount],
        "hour_of_day": [hour_of_day],
        "day_of_week": [day_of_week],
        "location": [location],
        "txn_type": [txn_type],
        "is_new_device": [is_new_device],
        "num_txn_last_1hr": [num_txn_last_1hr],
        "num_txn_last_24hr": [num_txn_last_24hr],
        "avg_amount_last_week": [avg_amount_last_week],
        "is_new_receiver": [is_new_receiver],
    }
    return pd.DataFrame(data)


def score_transaction(form_data):
    """Run the model and return (result_dict, features_dict)."""
    df = build_input_df(form_data)
    proba = float(model.predict_proba(df)[0][1])
    risk_score = int(round(proba * 100))

    if risk_score >= 80:
        decision = "BLOCK"
        badge_class = "danger"
    elif risk_score >= 50:
        decision = "FLAG"
        badge_class = "warning"
    else:
        decision = "ALLOW"
        badge_class = "success"

    result = {
        "proba": f"{proba * 100:.1f}",
        "risk_score": risk_score,
        "decision": decision,
        "badge_class": badge_class,
    }
    # df.iloc[0] is a Series with numpy types, convert to dict
    return result, df.iloc[0].to_dict()


# ---------- ROUTES ----------

@app.route("/", methods=["GET", "POST"])
def index():
    form_defaults = {
        "amount": "",
        "hour_of_day": "",
        "day_of_week": "",
        "location": "Delhi",
        "txn_type": "P2P",
        "is_new_device": "0",
        "is_new_receiver": "0",
        "num_txn_last_1hr": "0",
        "num_txn_last_24hr": "0",
        "avg_amount_last_week": "",
    }

    result = None

    if request.method == "POST":
        form_data = {**form_defaults, **request.form.to_dict()}
        result, features = score_transaction(form_data)

        # Prepare row for DB log (cast to basic Python types)
        log_row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "amount": float(features["amount"]),
            "hour_of_day": int(features["hour_of_day"]),
            "day_of_week": int(features["day_of_week"]),
            "location": str(features["location"]),
            "txn_type": str(features["txn_type"]),
            "is_new_device": int(features["is_new_device"]),
            "num_txn_last_1hr": int(features["num_txn_last_1hr"]),
            "num_txn_last_24hr": int(features["num_txn_last_24hr"]),
            "avg_amount_last_week": float(features["avg_amount_last_week"]),
            "is_new_receiver": int(features["is_new_receiver"]),
            "risk_score": int(result["risk_score"]),
            "decision": result["decision"],
        }
        log_transaction(log_row)

        form_defaults = form_data

    return render_template("index.html", form=form_defaults, result=result)


@app.route("/dashboard")
def dashboard():
    # No DB file yet
    if not DB_PATH.exists():
        return render_template("dashboard.html", empty=True)

    df = get_dataframe("SELECT * FROM transactions ORDER BY timestamp DESC")

    if df.empty:
        return render_template("dashboard.html", empty=True)

    total_txn = int(len(df))
    suspicious_mask = df["decision"].isin(["BLOCK", "FLAG"])
    fraud_txn = int(suspicious_mask.sum())
    fraud_rate = round((fraud_txn / total_txn) * 100, 1)

    # Fraud by city (top 5) - ensure pure Python ints
    city_grp = df[suspicious_mask].groupby("location").size().sort_values(ascending=False)
    city_top5 = city_grp.head(5)
    city_labels = [str(x) for x in city_top5.index.tolist()]
    city_data = [int(x) for x in city_top5.values.tolist()]  # <-- CAST TO INT

    # Fraud by hour - ensure pure Python ints
    hour_grp = df[suspicious_mask].groupby("hour_of_day").size().sort_index()
    hour_labels = [int(h) for h in hour_grp.index.tolist()]     # labels as ints
    hour_data = [int(x) for x in hour_grp.values.tolist()]      # <-- CAST TO INT

    # Recent suspicious
    recent_df = df[suspicious_mask].head(10)
    recent = recent_df.to_dict(orient="records")  # OK for table (no tojson)

    return render_template(
        "dashboard.html",
        empty=False,
        total_txn=total_txn,
        fraud_txn=fraud_txn,
        fraud_rate=fraud_rate,
        city_labels=city_labels,
        city_data=city_data,
        hour_labels=hour_labels,
        hour_data=hour_data,
        recent=recent,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON API for integration with other systems.

    Example payload:
    {
      "amount": 1500,
      "hour_of_day": 14,
      "day_of_week": 2,
      "location": "Delhi",
      "txn_type": "P2P",
      "is_new_device": 0,
      "num_txn_last_1hr": 1,
      "num_txn_last_24hr": 5,
      "avg_amount_last_week": 800,
      "is_new_receiver": 1
    }
    """
    data = request.get_json(force=True) or {}
    result, features = score_transaction(data)

    # Log API predictions as well
    log_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "amount": float(features["amount"]),
        "hour_of_day": int(features["hour_of_day"]),
        "day_of_week": int(features["day_of_week"]),
        "location": str(features["location"]),
        "txn_type": str(features["txn_type"]),
        "is_new_device": int(features["is_new_device"]),
        "num_txn_last_1hr": int(features["num_txn_last_1hr"]),
        "num_txn_last_24hr": int(features["num_txn_last_24hr"]),
        "avg_amount_last_week": float(features["avg_amount_last_week"]),
        "is_new_receiver": int(features["is_new_receiver"]),
        "risk_score": int(result["risk_score"]),
        "decision": result["decision"],
    }
    log_transaction(log_row)

    # Also make sure everything in response is JSON-serializable
    safe_features = {k: (int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                     for k, v in features.items()}

    return jsonify(
        {
            "result": result,
            "features": safe_features,
            "model_meta": {
                "auc": float(feature_info.get("auc")) if feature_info.get("auc") is not None else None,
                "model_type": "RandomForestClassifier",
            },
        }
    )


if __name__ == "__main__":
    init_db()
    app.run(debug=True)

