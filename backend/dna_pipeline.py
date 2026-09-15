import argparse
import csv
import math
import os
import re
from collections import Counter
from itertools import product

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils import CodonAdaptationIndex, gc_fraction

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


VALID_BASES = set("ACGTN")


COMMON_SCAR_MOTIFS = {
    "BsaI":  "GGTCTC",
    "BsmBI": "CGTCTC",
    "BbsI":  "GAAGAC",
    "SapI":  "GCTCTTC",
    "AarI":  "CACCTGC",
}

STOP_CODONS = {"TAA", "TAG", "TGA"}


# 1. VALIDATE FASTA


def validate_fasta(filepath):
    """
    Parse and validate a FASTA file.
    Returns (records, warnings). Raises on unrecoverable problems
    (file missing, unparseable, or completely empty).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        parsed = list(SeqIO.parse(filepath, "fasta"))
    except Exception as e:
        raise ValueError(f"Could not parse '{filepath}' as FASTA: {e}")

    if not parsed:
        raise ValueError(f"No sequences found in '{filepath}'. Is it a valid FASTA file?")

    records, warnings = [], []
    for rec in parsed:
        seq_str = str(rec.seq).upper()

        if len(seq_str) == 0:
            warnings.append(f"[{rec.id}] empty sequence — skipped.")
            continue

        invalid_chars = set(seq_str) - VALID_BASES
        if invalid_chars:
            warnings.append(
                f"[{rec.id}] contains non-standard base(s) {sorted(invalid_chars)} "
                f"— kept, but downstream features may be noisy."
            )

        if len(seq_str) < 30:
            warnings.append(f"[{rec.id}] very short ({len(seq_str)} bp) — features may be unreliable.")

        records.append(rec)

    if not records:
        raise ValueError("Every sequence in the file failed validation.")

    return records, warnings


# =========================================================
# 2. GC / AT CONTENT
# =========================================================

def calculate_gc_at(sequence: str) -> dict:
    sequence = sequence.upper()
    gc = gc_fraction(Seq(sequence)) * 100
    return {"gc_percent": round(gc, 2), "at_percent": round(100 - gc, 2)}


# =========================================================
# 3. K-MER FREQUENCY FEATURES (k = 3, 4)
# =========================================================

def get_kmer_frequencies(sequence: str, k: int) -> dict:
    sequence = sequence.upper()
    all_kmers = ["".join(p) for p in product("ACGT", repeat=k)]
    counts = dict.fromkeys(all_kmers, 0)
    total = 0
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if kmer in counts:
            counts[kmer] += 1
            total += 1
    if total == 0:
        return {kmer: 0.0 for kmer in all_kmers}
    return {kmer: count / total for kmer, count in counts.items()}


def kmer_summary(sequence: str, k: int) -> dict:
    """Compress the full k-mer vector into CSV-friendly summary stats."""
    freqs = get_kmer_frequencies(sequence, k)
    nonzero = [f for f in freqs.values() if f > 0]
    diversity = -sum(f * math.log2(f) for f in nonzero) if nonzero else 0.0
    top_kmer = max(freqs, key=freqs.get)
    return {
        f"kmer{k}_diversity": round(diversity, 3),
        f"kmer{k}_top": top_kmer,
        f"kmer{k}_top_freq": round(freqs[top_kmer], 4),
    }


# =========================================================
# 4. SHANNON ENTROPY
# =========================================================

def shannon_entropy(sequence: str) -> float:
    """Base-2 entropy of nucleotide composition. Max = 2.0 bits (fully random ACGT)."""
    sequence = sequence.upper()
    length = len(sequence)
    if length == 0:
        return 0.0
    counts = Counter(sequence)
    return round(-sum((c / length) * math.log2(c / length) for c in counts.values()), 4)


def windowed_entropy(sequence: str, window: int = 50, step: int = 25) -> list:
    seq = sequence.upper()
    out = []
    for i in range(0, max(len(seq) - window + 1, 1), step):
        w = seq[i:i + window]
        out.append({"start": i, "end": i + len(w), "entropy": shannon_entropy(w)})
    return out


# =========================================================
# 5. REPEAT / COMPLEXITY FEATURES
# =========================================================

def longest_tandem_repeat(sequence: str, max_unit: int = 6) -> dict:
    """Finds the longest simple tandem repeat (e.g. ATATAT..., CAGCAGCAG...)."""
    seq = sequence.upper()
    longest_len, longest_unit = 0, ""
    for unit_len in range(1, max_unit + 1):
        for m in re.finditer(r"(.{%d})\1{2,}" % unit_len, seq):
            hit_len = len(m.group(0))
            if hit_len > longest_len:
                longest_len, longest_unit = hit_len, m.group(1)
    return {"longest_repeat_len": longest_len, "longest_repeat_unit": longest_unit}


def linguistic_complexity(sequence: str, k: int = 4) -> float:
    """
    Ratio of distinct k-mers actually observed to the max possible.
    Close to 1.0 = highly varied sequence; close to 0 = repetitive/low-complexity.
    """
    seq = sequence.upper()
    n_windows = max(len(seq) - k + 1, 1)
    possible = min(4 ** k, n_windows)
    observed = len({seq[i:i + k] for i in range(len(seq) - k + 1)})
    return round(observed / possible, 4) if possible else 0.0


# =========================================================
# 6. CODING REGION (ORF) IDENTIFICATION
# =========================================================

def find_orfs(sequence: str, min_length: int = 100) -> list:
    """
    Scan all 6 reading frames (3 forward + 3 reverse-complement) for
    ORFs: ATG ... in-frame stop codon, length >= min_length nt.
    """
    seq = sequence.upper()
    orfs = []
    for strand, nuc in (("+", seq), ("-", str(Seq(seq).reverse_complement()))):
        for frame in range(3):
            i = frame
            while i < len(nuc) - 2:
                if nuc[i:i + 3] == "ATG":
                    j = i
                    found_stop = False
                    while j < len(nuc) - 2:
                        codon = nuc[j:j + 3]
                        if codon in STOP_CODONS:
                            found_stop = True
                            orf_len = j + 3 - i
                            if orf_len >= min_length:
                                orfs.append({
                                    "strand": strand, "frame": frame,
                                    "start": i, "end": j + 3, "length": orf_len,
                                })
                            break
                        j += 3
                    i = j if found_stop else i
                i += 3
    return orfs


# =========================================================
# 7. CODON-USAGE FEATURES (CAI, GC3) FOR DETECTED ORFs
# =========================================================

def build_cai_index(reference_sequences: list, table=None):
    if table is not None:
        return CodonAdaptationIndex(reference_sequences, table=table)
    return CodonAdaptationIndex(reference_sequences)


def codon_features_for_orfs(sequence: str, orfs: list, cai_index=None) -> dict:
    seq_fwd = sequence.upper()
    seq_rev = str(Seq(seq_fwd).reverse_complement())

    cais, gc3s = [], []
    for orf in orfs:
        src = seq_fwd if orf["strand"] == "+" else seq_rev
        sub = src[orf["start"]:orf["end"]]
        sub = sub[:len(sub) - (len(sub) % 3)]
        if len(sub) < 6:
            continue

        if cai_index is not None:
            try:
                cais.append(cai_index.calculate(sub))
            except Exception:
                pass

        third_pos = sub[2::3]
        if third_pos:
            gc3 = (third_pos.count("G") + third_pos.count("C")) / len(third_pos) * 100
            gc3s.append(gc3)

    return {
        "num_orfs": len(orfs),
        "mean_cai": round(sum(cais) / len(cais), 4) if cais else None,
        "mean_gc3": round(sum(gc3s) / len(gc3s), 2) if gc3s else None,
    }


# =========================================================
# 8. RESTRICTION SCAR / ASSEMBLY-JUNCTION PATTERN MATCHING
# =========================================================

def find_restriction_scars(sequence: str, motif_dict: dict = None) -> list:
    seq = sequence.upper()
    motif_dict = motif_dict or COMMON_SCAR_MOTIFS
    hits = []
    for name, motif in motif_dict.items():
        rev_comp = str(Seq(motif).reverse_complement())
        for pattern, strand in ((motif, "+"), (rev_comp, "-")):
            for m in re.finditer(pattern, seq):
                hits.append({
                    "enzyme_or_scar": name, "motif": pattern,
                    "strand": strand, "start": m.start(), "end": m.end(),
                })
    return hits


# =========================================================
# 9. SEQUENCE SIMILARITY ANALYSIS (all-vs-all, k-mer Jaccard)
# =========================================================

def _kmer_set(sequence: str, k: int = 6) -> set:
    seq = sequence.upper()
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def pairwise_similarity(records: list, k: int = 6) -> dict:
    """
    Fast, scalable similarity metric: Jaccard index over k=6 k-mer sets.
    (Swap in Bio.Align.PairwiseAligner for exact alignment identity if
    your dataset is small and you want alignment-based %identity instead.)
    Returns {seq_id: max_similarity_to_any_other_sequence}.
    """
    ids = [r.id for r in records]
    kmer_sets = {r.id: _kmer_set(str(r.seq), k) for r in records}
    max_sim = {i: 0.0 for i in ids}

    for a in range(len(ids)):
        for b in range(len(ids)):
            if a == b:
                continue
            sa, sb = kmer_sets[ids[a]], kmer_sets[ids[b]]
            union = sa | sb
            sim = len(sa & sb) / len(union) if union else 0.0
            max_sim[ids[a]] = max(max_sim[ids[a]], sim)

    return {k: round(v, 4) for k, v in max_sim.items()}


# =========================================================
# PER-SEQUENCE FEATURE ASSEMBLY
# =========================================================

def extract_all_features(record, cai_index=None) -> dict:
    seq = str(record.seq).upper()

    features = {"sequence_id": record.id, "length_bp": len(seq)}
    features.update(calculate_gc_at(seq))
    features.update(kmer_summary(seq, 3))
    features.update(kmer_summary(seq, 4))
    features["shannon_entropy"] = shannon_entropy(seq)
    features.update(longest_tandem_repeat(seq))
    features["linguistic_complexity"] = linguistic_complexity(seq)

    orfs = find_orfs(seq)
    features.update(codon_features_for_orfs(seq, orfs, cai_index))

    scars = find_restriction_scars(seq)
    features["num_restriction_scars"] = len(scars)
    features["scar_enzymes"] = ",".join(sorted({s["enzyme_or_scar"] for s in scars})) or "none"

    # kept for the visualization step, stripped out before CSV writing
    features["_orfs"] = orfs
    features["_scars"] = scars
    return features


# =========================================================
# CSV OUTPUT
# =========================================================

CSV_COLUMNS = [
    "sequence_id", "length_bp", "gc_percent", "at_percent",
    "kmer3_diversity", "kmer3_top", "kmer3_top_freq",
    "kmer4_diversity", "kmer4_top", "kmer4_top_freq",
    "shannon_entropy", "longest_repeat_len", "longest_repeat_unit",
    "linguistic_complexity", "num_orfs", "mean_cai", "mean_gc3",
    "num_restriction_scars", "scar_enzymes", "max_similarity_to_others",
]


def write_csv(all_features: list, similarity: dict, output_path: str):
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for feat in all_features:
            row = {col: feat.get(col, "") for col in CSV_COLUMNS}
            row["max_similarity_to_others"] = similarity.get(feat["sequence_id"], "")
            writer.writerow(row)


# =========================================================
# 10. VISUALIZATION — linear DNA map + GC track
# =========================================================

def plot_sequence_map(record, features: dict, output_path: str):
    seq = str(record.seq).upper()
    length = len(seq)

    fig, (ax_map, ax_gc) = plt.subplots(
        2, 1, figsize=(12, 4), gridspec_kw={"height_ratios": [1, 1.2]}
    )

    # --- top panel: linear DNA backbone with ORFs and restriction scars ---
    ax_map.set_xlim(0, length)
    ax_map.set_ylim(0, 3)
    ax_map.axis("off")
    ax_map.add_patch(mpatches.Rectangle((0, 1.4), length, 0.2, color="#cccccc"))
    ax_map.set_title(f"{record.id}  ({length} bp)", fontsize=11, loc="left")

    for orf in features["_orfs"]:
        color = "#4C72B0" if orf["strand"] == "+" else "#8172B2"
        y = 1.9 if orf["strand"] == "+" else 1.1
        ax_map.add_patch(mpatches.Rectangle(
            (orf["start"], y), orf["end"] - orf["start"], 0.25, color=color, alpha=0.7
        ))

    scar_colors = {}
    palette = ["#C44E52", "#DD8452", "#55A868", "#937860", "#CCB974"]
    for idx, name in enumerate(sorted({s["enzyme_or_scar"] for s in features["_scars"]})):
        scar_colors[name] = palette[idx % len(palette)]

    for scar in features["_scars"]:
        color = scar_colors[scar["enzyme_or_scar"]]
        ax_map.axvline(scar["start"], color=color, linewidth=2, ymin=0.3, ymax=0.7)
        ax_map.text(scar["start"], 2.5, scar["enzyme_or_scar"], rotation=90,
                     fontsize=7, color=color, ha="center", va="bottom")

    legend_handles = [mpatches.Patch(color="#4C72B0", label="ORF (+ strand)"),
                       mpatches.Patch(color="#8172B2", label="ORF (- strand)")]
    legend_handles += [mpatches.Patch(color=c, label=f"{n} scar") for n, c in scar_colors.items()]
    if legend_handles:
        ax_map.legend(handles=legend_handles, loc="upper right", fontsize=7, ncol=2, frameon=False)

    # --- bottom panel: GC content sliding window ---
    window, step = 50, 10
    positions, gc_vals = [], []
    for i in range(0, max(length - window + 1, 1), step):
        w = seq[i:i + window]
        positions.append(i + window / 2)
        gc_vals.append((w.count("G") + w.count("C")) / len(w) * 100)

    ax_gc.plot(positions, gc_vals, color="#4C72B0", linewidth=1.2)
    ax_gc.axhline(50, color="gray", linewidth=0.8, linestyle="--")
    ax_gc.set_xlim(0, length)
    ax_gc.set_ylabel("GC %")
    ax_gc.set_xlabel("Position (bp)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


# =========================================================
# PIPELINE ENTRY POINT
# =========================================================

def run_pipeline(fasta_paths, output_dir: str, reference_fasta: str = None, max_plots: int = 20):
    """
    fasta_paths: a single path (str) or a list of paths. Multiple files are
    read and pooled into one run, so you can hand it several .fasta uploads
    at once (e.g. synthetic_1.fasta, synthetic_2.fasta) and get one combined
    sequence_features.csv covering every sequence in every file.
    """
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(fasta_paths, str):
        fasta_paths = [fasta_paths]

    records = []
    for path in fasta_paths:
        file_records, warnings = validate_fasta(path)
        print(f"Validated {len(file_records)} sequence(s) from '{path}'.")
        for w in warnings:
            print("  WARNING:", w)
        records.extend(file_records)

    if not records:
        raise ValueError("No valid sequences found across the given file(s).")

    cai_index = None
    if reference_fasta:
        ref_records, ref_warnings = validate_fasta(reference_fasta)
        for w in ref_warnings:
            print("  REFERENCE WARNING:", w)
        cai_index = build_cai_index([str(r.seq) for r in ref_records])
        print(f"Built CAI reference index from {len(ref_records)} gene(s).")
    else:
        print("  No --reference provided: CAI will be reported as None. "
              "Pass a FASTA of highly-expressed genes for your host organism to enable it.")

    all_features = [extract_all_features(r, cai_index) for r in records]
    similarity = pairwise_similarity(records)

    csv_path = os.path.join(output_dir, "sequence_features.csv")
    write_csv(all_features, similarity, csv_path)
    print(f"Wrote {csv_path}")

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    for record, feat in zip(records[:max_plots], all_features[:max_plots]):
        plot_path = os.path.join(plots_dir, f"{record.id}_map.png")
        plot_sequence_map(record, feat, plot_path)
        print(f"Wrote {plot_path}")

    return csv_path, plots_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SynDNA bioinformatics feature extraction pipeline")
    parser.add_argument("--fasta", required=True, nargs="+",
                         help="Path to one or more input .fasta files (space-separated)")
    parser.add_argument("--output-dir", default="results", help="Directory to write outputs to")
    parser.add_argument("--reference", default=None, help="Optional FASTA of highly-expressed genes for CAI")
    parser.add_argument("--max-plots", type=int, default=20, help="Max number of sequences to visualize")
    args = parser.parse_args()

    run_pipeline(args.fasta, args.output_dir, args.reference, args.max_plots)