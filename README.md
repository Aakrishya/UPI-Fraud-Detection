# 🛡️ AI-Powered UPI Fraud Detection System
Real-Time Machine Learning Risk Scoring • Behavioural Analysis • Fraud Analytics Dashboard

## 📌 Overview
This project is a complete end-to-end UPI Fraud Detection System designed to mimic real-world fraud engines used by banks and payment apps. It analyzes transaction behavior, timing, device changes, and historical frequency to classify transactions as ALLOW, FLAG, or BLOCK.

## 🧰 Tech Stack
### Backend
- Python 3
- Flask
- SQLite
- Pandas, NumPy
- Scikit-learn
- Joblib

### Frontend
- HTML5
- Bootstrap 5
- Bootstrap Icons
- Jinja2
- Chart.js

### Machine Learning
- RandomForestClassifier
- OneHotEncoder + ColumnTransformer
- Pipeline-based preprocessing
- Synthetic dataset (20,000+ UPI transactions)

## 🎯 Purpose
To demonstrate how AI can detect fraud using:
- Behavioral analysis
- Timing & frequency patterns
- Device anomalies
- Receiver risk profiling
- ML-based risk scoring

## ⭐ Key Features
- Fraud prediction engine
- Real-time dashboard
- SQLite transaction logging
- API endpoint for prediction
- Beautiful responsive UI

## 🧠 Model Details
### Model Used
RandomForestClassifier (Scikit-Learn)

### Model Parameters
```
RandomForestClassifier(
    n_estimators=150,
    max_depth=None,
    random_state=42,
    class_weight="balanced"
)
```

### Why Random Forest?
- Handles mixed categorical + numeric data
- Detects nonlinear fraud patterns
- Robust for imbalanced data
- Stable and widely used in fintech

## 📊 Metrics Used
### 1. Confusion Matrix
- TP, TN, FP, FN breakdown

### 2. Classification Report
- Precision
- Recall
- F1-score

### 3. ROC-AUC Score
Industry standard for fraud detection:
```
roc_auc_score(y_test, y_proba)
```

## 🏗 Architecture
```
User Input → Flask Backend → ML Model → Rule Engine → SQLite Logging → Analytics Dashboard
```

## 📁 Folder Structure
```
upi-fraud-detection/
│
├── app.py
├── models/
│   ├── upi_fraud_model.pkl
│   └── feature_info.pkl
├── templates/
│   ├── index.html
│   └── dashboard.html
├── static/
├── generate_upi_dataset.py
├── train_upi_fraud_model.py
└── fraud_logs.db
```

## 🔮 Future Enhancements
- XGBoost / LightGBM models
- LSTM behavioral modeling
- Graph network fraud detection
- Device fingerprinting
- GPS/IP anomaly detection
- Heatmaps & trend visualization
- Cloud deployment (AWS/Render)

## ⚙ Installation
```
pip install -r requirements.txt
python train_upi_fraud_model.py
python app.py
```

## 🏁 Conclusion
This project brings together ML, backend development, frontend UI, fraud analytics, and real-time prediction—making it ideal for academic or real-world fintech learning.


