import os
import joblib
import pandas as pd


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "syndna_final_model.pkl"
)

FEATURE_PATH = os.path.join(
    os.path.dirname(__file__),
    "final_feature_columns.csv"
)


# Load model once when backend starts
model = joblib.load(MODEL_PATH)

# Exact feature order used during training
FEATURE_COLUMNS = pd.read_csv(FEATURE_PATH)["feature"].tolist()


def predict_anomaly(features: dict) -> dict:
    """
    Run Member 3's trained XGBoost model.
    """

    # Arrange features in the exact training order
    input_data = {
        feature: features.get(feature, 0.0)
        for feature in FEATURE_COLUMNS
    }

    X = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)

    # Probability of class 1 = synthetic/anomalous
    probabilities = model.predict_proba(X)[0]

    if len(probabilities) == 2:
        anomaly_score = float(probabilities[1])
    else:
        anomaly_score = float(probabilities[0])

    prediction = int(model.predict(X)[0])

    if anomaly_score >= 0.70:
        classification = "High"
    elif anomaly_score >= 0.40:
        classification = "Medium"
    else:
        classification = "Low"

    # Global model feature importance
    importance = model.feature_importances_

    top_indices = importance.argsort()[-10:][::-1]

    feature_importance = {
        FEATURE_COLUMNS[i]: float(importance[i])
        for i in top_indices
        if importance[i] > 0
    }

    return {
        "anomaly_score": anomaly_score,
        "prediction": prediction,
        "classification": classification,
        "feature_importance": feature_importance
    }