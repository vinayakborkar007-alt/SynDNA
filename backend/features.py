def extract_features(sequence: str, host: str = "unknown"):
    """
    Extract sequence features.

    This function will later be connected to Member 2's
    k-mer, CAI, GC-skew and restriction-scar features.
    """

    return {
        "kmer": 0.42,
        "cai": 0.31,
        "gc_skew": 0.18
    }
    