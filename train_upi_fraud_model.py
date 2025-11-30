# train_upi_fraud_model.py

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


def load_data(path="data/upi_transactions_dataset.csv"):
    return pd.read_csv(path)


def train_model():
    df = load_data()

    # Features and target
    X = df.drop(columns=["label", "sender_upi", "receiver_upi", "device_id"])
    y = df["label"]

    # Categorical and numeric features
    categorical_features = ["location", "txn_type", "is_new_device", "is_new_receiver"]
    numeric_features = [col for col in X.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="passthrough",
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", rf),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {auc:.4f}")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "upi_fraud_model.pkl"
    joblib.dump(clf, model_path)
    print(f"\n✅ Model saved to {model_path}")

    feature_info = {
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "auc": float(auc),
    }
    joblib.dump(feature_info, models_dir / "feature_info.pkl")
    print("✅ Feature info saved to models/feature_info.pkl")


if __name__ == "__main__":
    train_model()

