from features import extract_features


def run_prediction(sequence: str, host: str = "unknown"):
    """
    Run the SynDNA prediction pipeline.

    Current flow:
    sequence -> feature extraction -> dummy prediction

    The feature extractor will later contain Member 2's
    actual k-mer, CAI, GC-skew and restriction-scar features.
    """

    features = extract_features(sequence, host)

    return {
        "anomaly_score": 0.78,
        "classification": "High",
        "suspicious_regions": [],
        "feature_importance": features
    }