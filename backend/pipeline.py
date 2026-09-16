"""End-to-end XGBoost inference and model-driven suspicious-region mapping."""
from __future__ import annotations

from .features import clean_sequence, extract_features
from .model import predict

WINDOW_SIZE = 500
STEP_SIZE = 100
REGION_THRESHOLD = 0.65


def predict_sequence(raw_sequence: str) -> dict:
    seq = clean_sequence(raw_sequence)
    if len(seq) < 20:
        raise ValueError("Sequence must contain at least 20 valid bases")
    features = extract_features(seq)
    result = predict(features)
    result.update({"features": features, "sequence_length": len(seq)})
    return result


def map_suspicious_regions(raw_sequence: str, window_size: int = WINDOW_SIZE, step_size: int = STEP_SIZE, threshold: float = REGION_THRESHOLD) -> dict:
    seq = clean_sequence(raw_sequence)
    n = len(seq)
    if n < 20:
        raise ValueError("Sequence must contain at least 20 valid bases")

    # Short sequence: score the entire input with exactly the same model.
    if n <= window_size:
        score = predict_sequence(seq)["anomaly_score"]
        region = {"start": 1, "end": n, "score": round(score, 4), "score_pct": round(score * 100, 1)} if score >= threshold else None
        return {"start": region["start"] if region else None,
                "end": region["end"] if region else None,
                "score": round(score, 4), "score_pct": round(score * 100, 1),
                "threshold": threshold,
                "windows": [{"start": 1, "end": n, "score": round(score, 4)}]}

    windows = []
    starts = list(range(0, n - window_size + 1, step_size))
    if not starts or starts[-1] != n - window_size:
        starts.append(n - window_size)

    for start0 in starts:
        end0 = start0 + window_size
        score = predict_sequence(seq[start0:end0])["anomaly_score"]
        windows.append({"start": start0 + 1, "end": end0, "score": score})

    flagged = [w for w in windows if w["score"] >= threshold]
    if not flagged:
        best = max(windows, key=lambda w: w["score"])
        return {"start": None, "end": None, "score": round(best["score"], 4), "score_pct": round(best["score"] * 100, 1),
                "threshold": threshold, "windows": [{**w, "score": round(w["score"], 4)} for w in windows]}

    merged = []
    for w in flagged:
        if not merged or w["start"] > merged[-1]["end"] + step_size:
            merged.append({"start": w["start"], "end": w["end"], "score": w["score"]})
        else:
            merged[-1]["end"] = max(merged[-1]["end"], w["end"])
            merged[-1]["score"] = max(merged[-1]["score"], w["score"])

    strongest = max(merged, key=lambda r: r["score"])
    return {
        "start": int(strongest["start"]), "end": int(strongest["end"]),
        "score": round(strongest["score"], 4), "score_pct": round(strongest["score"] * 100, 1),
        "threshold": threshold,
        "regions": [{"start": int(r["start"]), "end": int(r["end"]), "score": round(r["score"], 4), "score_pct": round(r["score"] * 100, 1)} for r in merged],
        "windows": [{**w, "score": round(w["score"], 4)} for w in windows],
    }


def analyze(raw_sequence: str, host: str = "ecoli") -> dict:
    seq = clean_sequence(raw_sequence)
    prediction = predict_sequence(seq)
    region = map_suspicious_regions(seq, threshold=REGION_THRESHOLD)
    return {
        **prediction,
        "verdict": prediction["classification"],
        "host": host,
        "flagged_region": region,
        "model": {"name": "XGBoost", "training_rows": 40, "natural_rows": 20, "synthetic_rows": 20},
    }
