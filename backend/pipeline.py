from features import extract_features
from model import predict_anomaly


def run_prediction(sequence: str, host: str = "unknown") -> dict:
    """
    Complete SynDNA prediction pipeline:

    DNA sequence
        ↓
    Feature extraction
        ↓
    Member 3 XGBoost model
        ↓
    Prediction result
    """

    features = extract_features(sequence, host)

    prediction = predict_anomaly(features)

    return {
        "anomaly_score": prediction["anomaly_score"],
        "classification": prediction["classification"],
        "prediction": prediction["prediction"],
        "suspicious_regions": [],
        "feature_importance": prediction["feature_importance"]
    }