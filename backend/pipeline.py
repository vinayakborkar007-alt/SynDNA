def run_prediction(sequence: str, host: str = "unknown"):
    """
    Temporary prediction pipeline.

    Later this will connect:
    sequence -> feature extraction -> ML model -> analysis
    """

    return {
        "anomaly_score": 0.78,
        "classification": "High",
        "suspicious_regions": [],
        "feature_importance": {
            "kmer": 0.42,
            "cai": 0.31,
            "gc_skew": 0.18
        }
    }