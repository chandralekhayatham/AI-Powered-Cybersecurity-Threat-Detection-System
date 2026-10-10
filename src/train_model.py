"""Train and evaluate a defensive ML classifier on synthetic network telemetry."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import plot_tree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES, MODEL_FEATURES, TARGET

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "network_traffic.csv"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"

def build_pipeline():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)])
    return Pipeline([("preprocess", prep), ("classifier", RandomForestClassifier(n_estimators=160, class_weight="balanced", random_state=42, min_samples_leaf=2, n_jobs=-1))])

def main():
    if not DATA.exists():
        raise SystemExit("Dataset not found. Run: python data/generate_dataset.py")
    MODELS.mkdir(exist_ok=True); OUTPUTS.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    X, y = df[MODEL_FEATURES], df[TARGET].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = build_pipeline()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    labels = sorted(y.unique())
    metrics = {"dataset_records": int(len(df)), "train_records": int(len(X_train)), "test_records": int(len(X_test)), "accuracy": round(float(accuracy_score(y_test, pred)), 4), "precision_macro": round(float(precision_score(y_test, pred, average="macro", zero_division=0)), 4), "recall_macro": round(float(recall_score(y_test, pred, average="macro", zero_division=0)), 4), "f1_macro": round(float(f1_score(y_test, pred, average="macro", zero_division=0)), 4), "labels": labels, "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0), "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(), "note": "Results are from a synthetic educational dataset; they do not represent production detection performance."}
    joblib.dump(model, MODELS / "threat_detector.joblib")
    (OUTPUTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # Confusion matrix image for README/portfolio evidence.
    cm = confusion_matrix(y_test, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix — Synthetic Demo Dataset")
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=20); ax.set_yticks(range(len(labels)), labels=labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout(); fig.savefig(OUTPUTS / "confusion_matrix.png", dpi=150); plt.close(fig)
    # Feature importance is extracted after the preprocessing pipeline.
    names = model.named_steps["preprocess"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_
    top = sorted(zip(names, importances), key=lambda z: z[1], reverse=True)[:12]
    fig, ax = plt.subplots(figsize=(8, 5)); ax.barh([x[0] for x in top][::-1], [float(x[1]) for x in top][::-1])
    ax.set_title("Top Model Feature Importances (Synthetic Demo)"); ax.set_xlabel("Importance"); fig.tight_layout()
    fig.savefig(OUTPUTS / "feature_importance.png", dpi=150); plt.close(fig)
    print("Model training completed.")
    print(f"Records: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Accuracy: {metrics['accuracy']:.3f} | Precision (macro): {metrics['precision_macro']:.3f} | Recall (macro): {metrics['recall_macro']:.3f} | F1 (macro): {metrics['f1_macro']:.3f}")
    print(f"Saved model: {MODELS / 'threat_detector.joblib'}")
    print(f"Saved metrics and charts in: {OUTPUTS}")
    print("Reminder: metrics are illustrative results on synthetic data, not proof of real-world effectiveness.")

if __name__ == "__main__": main()
