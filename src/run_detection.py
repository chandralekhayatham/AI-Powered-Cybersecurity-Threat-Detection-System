"""Run offline batch predictions against the generated demo telemetry."""
from pathlib import Path
import joblib
import pandas as pd
from src.features import MODEL_FEATURES
ROOT = Path(__file__).resolve().parents[1]
def main():
    model_path = ROOT / "models" / "threat_detector.joblib"
    data_path = ROOT / "data" / "network_traffic.csv"
    if not model_path.exists(): raise SystemExit("Model missing. Run: python -m src.train_model")
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)
    df["predicted_label"] = model.predict(df[MODEL_FEATURES])
    df["threat_probability"] = model.predict_proba(df[MODEL_FEATURES]).max(axis=1).round(4)
    df["alert_level"] = df.apply(lambda r: "HIGH" if r.predicted_label != "normal" and r.threat_probability >= 0.75 else ("REVIEW" if r.predicted_label != "normal" else "INFO"), axis=1)
    out = ROOT / "outputs" / "predictions.csv"; out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Predictions saved to {out}")
    print("Predicted labels:"); print(df["predicted_label"].value_counts().to_string())
    print("Alert levels:"); print(df["alert_level"].value_counts().to_string())
    print("Sample flagged rows:"); print(df[df.alert_level != "INFO"][["event_id", "predicted_label", "threat_probability", "alert_level"]].head(10).to_string(index=False))
if __name__ == "__main__": main()
