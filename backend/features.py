import itertools
from collections import Counter

import pandas as pd

from dna_pipeline import (
    calculate_gc_at,
    shannon_entropy,
    longest_tandem_repeat,
    find_orfs,
)


BASES = "ACGT"


def _kmer_frequencies(sequence: str, k: int) -> dict:
    """Return normalized frequency for every possible k-mer."""
    counts = Counter(
        sequence[i:i + k]
        for i in range(len(sequence) - k + 1)
    )

    total = sum(counts.values())

    all_kmers = [
        "".join(x)
        for x in itertools.product(BASES, repeat=k)
    ]

    if total == 0:
        return {kmer: 0.0 for kmer in all_kmers}

    return {
        kmer: counts.get(kmer, 0) / total
        for kmer in all_kmers
    }


def _codon_frequencies(sequence: str) -> dict:
    """Return normalized frequency of codons in the sequence."""
    counts = Counter(
        sequence[i:i + 3]
        for i in range(0, len(sequence) - 2, 3)
        if set(sequence[i:i + 3]).issubset(set(BASES))
    )

    total = sum(counts.values())

    codons = [
        "".join(x)
        for x in itertools.product(BASES, repeat=3)
    ]

    if total == 0:
        return {codon: 0.0 for codon in codons}

    return {
        codon: counts.get(codon, 0) / total
        for codon in codons
    }


def _gc_position_percentages(sequence: str):
    """GC percentage at codon positions 1, 2 and 3."""
    values = []

    for position in range(3):
        bases = sequence[position::3]

        if not bases:
            values.append(0.0)
        else:
            gc = sum(b in "GC" for b in bases)
            values.append((gc / len(bases)) * 100)

    return values


def _codon_bias_score(sequence: str) -> float:
    """
    Simple codon-bias measure.

    Measures how strongly the most frequent codons dominate
    their respective synonymous codon groups.
    """
    codons = [
        sequence[i:i + 3]
        for i in range(0, len(sequence) - 2, 3)
        if set(sequence[i:i + 3]).issubset(set(BASES))
    ]

    if not codons:
        return 0.0

    counts = Counter(codons)

    # Synonymous codon groups
    groups = [
        ["TTT", "TTC"],
        ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
        ["ATT", "ATC", "ATA"],
        ["ATG"],
        ["GTT", "GTC", "GTA", "GTG"],
        ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
        ["CCT", "CCC", "CCA", "CCG"],
        ["ACT", "ACC", "ACA", "ACG"],
        ["GCT", "GCC", "GCA", "GCG"],
        ["TAT", "TAC"],
        ["CAT", "CAC"],
        ["CAA", "CAG"],
        ["AAT", "AAC"],
        ["AAA", "AAG"],
        ["GAT", "GAC"],
        ["GAA", "GAG"],
        ["TGT", "TGC"],
        ["TGG"],
        ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
        ["GGT", "GGC", "GGA", "GGG"],
        ["ACT", "ACC", "ACA", "ACG"],
    ]

    scores = []

    for group in groups:
        total = sum(counts[c] for c in group)

        if total == 0 or len(group) == 1:
            continue

        maximum = max(counts[c] for c in group)
        scores.append(maximum / total)

    return sum(scores) / len(scores) if scores else 0.0


def _codon_adaptation_index(sequence: str) -> float:
    """
    Deterministic codon adaptation proxy.

    Uses the relative frequency of the most preferred codon
    within each synonymous codon family.
    """
    codons = [
        sequence[i:i + 3]
        for i in range(0, len(sequence) - 2, 3)
        if set(sequence[i:i + 3]).issubset(set(BASES))
    ]

    if not codons:
        return 0.0

    counts = Counter(codons)

    families = [
        ["TTT", "TTC"],
        ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
        ["ATT", "ATC", "ATA"],
        ["GTT", "GTC", "GTA", "GTG"],
        ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
        ["CCT", "CCC", "CCA", "CCG"],
        ["ACT", "ACC", "ACA", "ACG"],
        ["GCT", "GCC", "GCA", "GCG"],
        ["TAT", "TAC"],
        ["CAT", "CAC"],
        ["CAA", "CAG"],
        ["AAT", "AAC"],
        ["AAA", "AAG"],
        ["GAT", "GAC"],
        ["GAA", "GAG"],
        ["TGT", "TGC"],
        ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
        ["GGT", "GGC", "GGA", "GGG"],
    ]

    values = []

    for family in families:
        total = sum(counts[c] for c in family)

        if total == 0:
            continue

        maximum = max(counts[c] for c in family)

        values.append(maximum / total)

    return sum(values) / len(values) if values else 0.0


def extract_features(sequence: str, host: str = "unknown") -> dict:
    """
    Convert a DNA sequence into the 377 features expected by
    Member 3's trained XGBoost model.
    """

    sequence = sequence.strip().upper()

    if not sequence:
        raise ValueError("Sequence is empty")

    invalid = set(sequence) - set(BASES)

    if invalid:
        raise ValueError(
            f"Invalid DNA characters: {''.join(sorted(invalid))}"
        )

    length = len(sequence)

    # Basic composition
    a = sequence.count("A")
    t = sequence.count("T")
    c = sequence.count("C")
    g = sequence.count("G")

    gc_pct = ((g + c) / length) * 100 if length else 0.0

    gc_skew = (
        (g - c) / (g + c)
        if (g + c) else 0.0
    )

    at_skew = (
        (a - t) / (a + t)
        if (a + t) else 0.0
    )

    # GC position percentages
    gc1, gc2, gc3 = _gc_position_percentages(sequence)

    # ORFs
    orfs = find_orfs(sequence)

    orf_lengths = [
        len(orf)
        for orf in orfs
        if hasattr(orf, "__len__")
    ]

    average_orf_length = (
        sum(orf_lengths) / len(orf_lengths)
        if orf_lengths else 0.0
    )

    max_orf_length = max(orf_lengths) if orf_lengths else 0.0

    coding_fraction = (
        sum(orf_lengths) / length
        if length else 0.0
    )

    # Homopolymers
    homopolymer_lengths = []

    current = 1

    for i in range(1, length):
        if sequence[i] == sequence[i - 1]:
            current += 1
        else:
            homopolymer_lengths.append(current)
            current = 1

    if length:
        homopolymer_lengths.append(current)

    max_homopolymer = max(homopolymer_lengths) if homopolymer_lengths else 0
    mean_homopolymer = (
        sum(homopolymer_lengths) / len(homopolymer_lengths)
        if homopolymer_lengths else 0.0
    )

    c_homopolymer_count = 0
    current_c = 0

    for base in sequence:
        if base == "C":
            current_c += 1
        else:
            if current_c > 1:
                c_homopolymer_count += 1
            current_c = 0

    if current_c > 1:
        c_homopolymer_count += 1

    # Repeats
    repeats = longest_tandem_repeat(sequence)

    if isinstance(repeats, list):
        repeat_density = (
            sum(len(str(x)) for x in repeats) / length
            if length else 0.0
        )
    else:
        repeat_density = 0.0

    # Base feature dictionary
    features = {
        "length_bp": length,
        "gc_content_pct": gc_pct,
        "n_content_pct": 0.0,
        "a_frequency": a / length,
        "t_frequency": t / length,
        "gc_skew": gc_skew,
        "at_skew": at_skew,
        "sequence_entropy": shannon_entropy(sequence),
        "homopolymer_max_length": max_homopolymer,
        "homopolymer_mean_length": mean_homopolymer,
        "homopolymer_C_count": c_homopolymer_count,
        "repeat_density": repeat_density,
        "coding_fraction": coding_fraction,
        "average_orf_length": average_orf_length,
        "max_orf_length": max_orf_length,
        "codon_adaptation_index": _codon_adaptation_index(sequence),
        "codon_bias_score": _codon_bias_score(sequence),
        "gc1_pct": gc1,
        "gc2_pct": gc2,
        "gc3_pct": gc3,
    }

    # k = 2
    for kmer, value in _kmer_frequencies(sequence, 2).items():
        features[f"kmer_{kmer}"] = value

    # k = 3
    for kmer, value in _kmer_frequencies(sequence, 3).items():
        features[f"kmer_{kmer}"] = value

    # k = 4
    for kmer, value in _kmer_frequencies(sequence, 4).items():
        features[f"kmer_{kmer}"] = value

    # Codons
    for codon, value in _codon_frequencies(sequence).items():
        features[f"codon_{codon}"] = value

    # Load exact feature order used by Member 3
    columns = pd.read_csv("final_feature_columns.csv")["feature"].tolist()

    missing = [column for column in columns if column not in features]

    if missing:
        raise ValueError(
            f"Missing model features: {missing}"
        )

    # Return ONLY the exact 377 model features, in exact order
    return {
        column: float(features[column])
        for column in columns
    }