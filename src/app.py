"""Local Flask dashboard and prediction API. Never captures live traffic or contacts hosts."""
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request, render_template
from src.features import MODEL_FEATURES
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "threat_detector.joblib"
DATA_PATH = ROOT / "data" / "network_traffic.csv"
app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))

def get_model():
    if not MODEL_PATH.exists(): return None
    return joblib.load(MODEL_PATH)

def dashboard_summary():
    if not DATA_PATH.exists(): return {"dataset_records": 0, "message": "Generate the dataset first."}
    df = pd.read_csv(DATA_PATH)
    return {"dataset_records": int(len(df)), "label_counts": {str(k): int(v) for k,v in df["label"].value_counts().items()}, "attack_counts": {str(k): int(v) for k,v in df.loc[df.label != "normal", "attack_type"].value_counts().items()}, "protocol_counts": {str(k): int(v) for k,v in df["protocol"].value_counts().items()}, "sample": df.head(8).to_dict(orient="records"), "dataset_notice": "Synthetic educational telemetry only; no live network data."}

@app.get("/")
def index(): return render_template("index.html")

@app.get("/api/health")
def health(): return jsonify({"status": "ok", "model_ready": MODEL_PATH.exists(), "data_ready": DATA_PATH.exists(), "mode": "offline synthetic simulation"})

@app.get("/api/summary")
def summary(): return jsonify(dashboard_summary())

@app.post("/api/predict")
def predict():
    model = get_model()
    if model is None: return jsonify({"error": "Model not trained. Run: python -m src.train_model"}), 503
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict): return jsonify({"error": "Send a JSON object containing the required feature fields."}), 400
    missing = [c for c in MODEL_FEATURES if c not in payload]
    if missing: return jsonify({"error": "Missing required fields", "missing": missing}), 400
    try:
        row = {c: [payload[c]] for c in MODEL_FEATURES}
        X = pd.DataFrame(row)
        for col in ["duration_seconds", "src_bytes", "dst_bytes", "failed_logins", "request_frequency", "connection_count", "packet_size", "unique_ports_contacted"]: X[col] = pd.to_numeric(X[col], errors="raise")
        if (X.select_dtypes(include="number") < 0).any().any(): return jsonify({"error": "Numeric features must be non-negative."}), 400
        label = str(model.predict(X)[0]); probabilities = model.predict_proba(X)[0]
        classes = list(model.classes_); confidence = float(max(probabilities))
        level = "HIGH" if label != "normal" and confidence >= 0.75 else ("REVIEW" if label != "normal" else "INFO")
        return jsonify({"predicted_label": label, "model_confidence": round(confidence, 4), "alert_level": level, "interpretation": "Suspicious pattern predicted; review supporting telemetry. This is not proof of compromise." if label != "normal" else "Model classified this sample as normal; this does not guarantee it is safe.", "mode": "synthetic/offline demo"})
    except (ValueError, TypeError) as exc: return jsonify({"error": f"Invalid feature value: {exc}"}), 400

if __name__ == "__main__":
    print("Starting local synthetic telemetry dashboard: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
