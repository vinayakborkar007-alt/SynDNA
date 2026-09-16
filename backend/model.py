"""Load and run the XGBoost model trained on the 40-row dataset."""
from __future__ import annotations

from pathlib import Path
import json
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent
ARTIFACTS = BASE / "artifacts"
MODEL_PATH = ARTIFACTS / "xgboost_model_40.pkl"
SCHEMA_PATH = ARTIFACTS / "final_feature_columns.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
if not SCHEMA_PATH.exists():
    raise FileNotFoundError(f"Missing feature schema: {SCHEMA_PATH}")

MODEL = joblib.load(MODEL_PATH)
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
FEATURE_COLUMNS = SCHEMA["features"]


def predict(features: dict) -> dict:
    X = pd.DataFrame([[float(features.get(c, 0.0)) for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    probability = float(MODEL.predict_proba(X)[0, 1])
    label = int(probability >= 0.50)
    return {
        "label": label,
        "classification": "SYNTHETIC" if label else "NATURAL",
        "anomaly_score": probability,
        "anomaly_score_pct": round(probability * 100.0, 1),
    }
