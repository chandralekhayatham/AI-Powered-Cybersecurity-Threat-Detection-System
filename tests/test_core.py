import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.features import MODEL_FEATURES
from data.generate_dataset import generate_dataset
from src.train_model import build_pipeline

def test_generator_has_expected_columns_and_safe_label():
    df = generate_dataset(100, seed=7)
    assert len(df) == 100
    assert set(MODEL_FEATURES).issubset(df.columns)
    assert set(df.label.unique()).issubset({"normal", "dos_like", "brute_force_like", "scan_like"})
    assert set(df.data_source) == {"SYNTHETIC_DEMO_ONLY"}

def test_features_are_nonnegative():
    df = generate_dataset(100, seed=9)
    numeric = [c for c in MODEL_FEATURES if c != "protocol"]
    assert (df[numeric] >= 0).all().all()

def test_pipeline_can_train_and_predict():
    df = generate_dataset(300, seed=12)
    model = build_pipeline()
    model.fit(df[MODEL_FEATURES], df.label)
    pred = model.predict(df[MODEL_FEATURES].head(5))
    assert len(pred) == 5
    assert set(pred).issubset(set(df.label.unique()))
