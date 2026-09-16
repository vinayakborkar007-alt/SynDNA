"""Feature extraction used by the 40-row SynDNA XGBoost model.

These are the same 17 feature columns used to train xgboost_model_40.pkl.
The extractor intentionally uses only sequence-derived quantities so the API
can score a new FASTA sequence and can reuse the exact extractor for windows.
"""
from __future__ import annotations

from collections import Counter
import math
import re
from typing import Dict, Tuple

FEATURE_COLUMNS = [
    "gc_content_pct", "at_content_pct", "n_content_pct",
    "a_frequency", "t_frequency", "c_frequency", "g_frequency",
    "gc_skew", "at_skew", "sequence_entropy",
    "homopolymer_max_length", "homopolymer_mean_length", "repeat_density",
    "low_complexity_fraction", "gc1_pct", "gc2_pct", "gc3_pct",
]


def clean_sequence(raw: str) -> str:
    lines = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        lines.append(line)
    return re.sub(r"[^ATGCN]", "", "".join(lines).upper())


def entropy(seq: str) -> float:
    if not seq:
        return 0.0
    counts = Counter(seq)
    n = len(seq)
    return float(-sum((c / n) * math.log2(c / n) for c in counts.values() if c))


def homopolymer_stats(seq: str) -> Tuple[int, float, int]:
    if not seq:
        return 0, 0.0, 0
    runs = []
    start = 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[start]:
            length = i - start
            if length >= 3:
                runs.append(length)
            start = i
    if not runs:
        return 0, 0.0, 0
    return max(runs), float(sum(runs) / len(runs)), len(runs)


def low_complexity_fraction(seq: str, window: int = 100, threshold: float = 1.65) -> float:
    if not seq:
        return 0.0
    if len(seq) < window:
        return 1.0 if entropy(seq) < threshold else 0.0
    low = 0
    total = 0
    for start in range(0, len(seq) - window + 1, window):
        total += 1
        if entropy(seq[start:start + window]) < threshold:
            low += 1
    return low / total if total else 0.0


def gc_by_frame(seq: str) -> Tuple[float, float, float]:
    out = []
    for frame in range(3):
        part = seq[frame::3]
        out.append(100.0 * sum(b in "GC" for b in part) / len(part) if part else 0.0)
    return tuple(out)


def extract_features(raw: str) -> Dict[str, float]:
    seq = clean_sequence(raw)
    if not seq:
        raise ValueError("No valid DNA bases found in sequence")

    n = len(seq)
    counts = Counter(seq)
    a, t, c, g = (counts.get(x, 0) for x in "ATCG")
    unknown = counts.get("N", 0)
    gc = g + c
    at = a + t

    hp_max, hp_mean, hp_count = homopolymer_stats(seq)
    gc1, gc2, gc3 = gc_by_frame(seq)

    return {
        "gc_content_pct": 100.0 * gc / n,
        "at_content_pct": 100.0 * at / n,
        "n_content_pct": 100.0 * unknown / n,
        "a_frequency": a / n,
        "t_frequency": t / n,
        "c_frequency": c / n,
        "g_frequency": g / n,
        "gc_skew": (g - c) / gc if gc else 0.0,
        "at_skew": (a - t) / at if at else 0.0,
        "sequence_entropy": entropy(seq),
        "homopolymer_max_length": hp_max,
        "homopolymer_mean_length": hp_mean,
        "repeat_density": (hp_count / n) * 1000.0,
        "low_complexity_fraction": low_complexity_fraction(seq),
        "gc1_pct": gc1,
        "gc2_pct": gc2,
        "gc3_pct": gc3,
    }
