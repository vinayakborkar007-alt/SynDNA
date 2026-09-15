from features import extract_features
from model import predict_anomaly


def run_prediction(sequence: str, host: str = "unknown"):
    """
    Run the SynDNA prediction pipeline.

    Flow:
    sequence -> feature extraction -> ML prediction
    """

    features = extract_features(sequence, host)

    prediction = predict_anomaly(features)

    return {
        "anomaly_score": prediction["anomaly_score"],
        "classification": prediction["classification"],
        "suspicious_regions": [],
        "feature_importance": features
    }