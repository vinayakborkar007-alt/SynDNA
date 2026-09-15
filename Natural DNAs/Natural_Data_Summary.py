import os
import math
import re
import itertools
import pandas as pd
import numpy as np
from Bio import SeqIO

# Define nucleotide and codon alphabets
NUCLEOTIDES = ['A', 'C', 'G', 'T']
KMERS_2 = [''.join(p) for p in itertools.product(NUCLEOTIDES, repeat=2)]
KMERS_3 = [''.join(p) for p in itertools.product(NUCLEOTIDES, repeat=3)]
KMERS_4 = [''.join(p) for p in itertools.product(NUCLEOTIDES, repeat=4)]
CODONS = [''.join(p) for p in itertools.product(NUCLEOTIDES, repeat=3)]

def calculate_entropy(seq):
    length = len(seq)
    if length == 0:
        return 0.0
    entropy = 0.0
    for base in NUCLEOTIDES:
        count = seq.count(base)
        if count > 0:
            p = count / length
            entropy -= p * math.log2(p)
    return round(entropy, 4)

def find_orfs(seq, min_length=100):
    start_codon = "ATG"
    stop_codons = {"TAA", "TAG", "TGA"}
    orfs = []
    
    for frame in range(3):
        i = frame
        while i < len(seq) - 2:
            if seq[i:i+3] == start_codon:
                for j in range(i + 3, len(seq) - 2, 3):
                    codon = seq[j:j+3]
                    if codon in stop_codons:
                        orf_len = (j + 3) - i
                        if orf_len >= min_length:
                            orfs.append(orf_len)
                        i = j
                        break
            i += 3
    return orfs

def process_fasta_file(filepath, filename):
    records = []
    
    # Class 0: Natural Genomes
    is_synthetic = 0
    organism = "Bacterial Genome"
    seq_type = "Natural Chromosome"
    
    for record in SeqIO.parse(filepath, "fasta"):
        seq = str(record.seq).upper()
        length = len(seq)
        if length == 0:
            continue

        # Basic Sequence Metrics
        cnt_a, cnt_c, cnt_g, cnt_t = seq.count('A'), seq.count('C'), seq.count('G'), seq.count('T')
        cnt_n = seq.count('N')
        
        a_freq = cnt_a / length
        c_freq = cnt_c / length
        g_freq = cnt_g / length
        t_freq = cnt_t / length
        
        gc_pct = (cnt_g + cnt_c) / length * 100
        at_pct = (cnt_a + cnt_t) / length * 100
        n_pct = cnt_n / length * 100
        
        # Skews
        gc_skew = (cnt_g - cnt_c) / (cnt_g + cnt_c) if (cnt_g + cnt_c) > 0 else 0.0
        at_skew = (cnt_a - cnt_t) / (cnt_a + cnt_t) if (cnt_a + cnt_t) > 0 else 0.0

        # Homopolymers (run length >= 3)
        homopolymers = re.findall(r'A{3,}|C{3,}|G{3,}|T{3,}', seq)
        homo_count = len(homopolymers)
        homo_lens = [len(h) for h in homopolymers]
        homo_max = max(homo_lens) if homo_lens else 0
        homo_mean = np.mean(homo_lens) if homo_lens else 0.0
        
        homo_a = sum(1 for h in homopolymers if h[0] == 'A')
        homo_t = sum(1 for h in homopolymers if h[0] == 'T')
        homo_c = sum(1 for h in homopolymers if h[0] == 'C')
        homo_g = sum(1 for h in homopolymers if h[0] == 'G')

        # Repeats & Low Complexity
        low_comp_matches = [m.group() for m in re.finditer(r'(.)\1{2,}', seq)]
        repeat_count = len(low_comp_matches)
        repeat_density = repeat_count / (length / 1000)  # per kb
        low_complexity_fraction = sum(len(m) for m in low_comp_matches) / length

        # ORF & Coding Region Features
        orfs = find_orfs(seq)
        orf_count = len(orfs)
        total_coding_bp = sum(orfs)
        coding_fraction = total_coding_bp / length
        avg_orf_len = np.mean(orfs) if orfs else 0.0
        max_orf_len = max(orfs) if orfs else 0

        # Codon Positions & CAI/Bias Approximations
        codons_in_seq = [seq[i:i+3] for i in range(0, length - 2, 3) if len(seq[i:i+3]) == 3]
        gc1_pct = np.mean([1 if c[0] in 'GC' else 0 for c in codons_in_seq]) * 100 if codons_in_seq else 0
        gc2_pct = np.mean([1 if c[1] in 'GC' else 0 for c in codons_in_seq]) * 100 if codons_in_seq else 0
        gc3_pct = np.mean([1 if c[2] in 'GC' else 0 for c in codons_in_seq]) * 100 if codons_in_seq else 0

        # K-mer Frequencies
        k2_counts = {f"kmer_{k}": seq.count(k) / (length - 1) if length > 1 else 0 for k in KMERS_2}
        k3_counts = {f"kmer_{k}": seq.count(k) / (length - 2) if length > 2 else 0 for k in KMERS_3}
        k4_counts = {f"kmer_{k}": seq.count(k) / (length - 3) if length > 3 else 0 for k in KMERS_4}

        # Codon Frequencies
        c_total = len(codons_in_seq)
        codon_counts = {f"codon_{c}": codons_in_seq.count(c) / c_total if c_total > 0 else 0 for c in CODONS}
        codon_bias = np.var(list(codon_counts.values())) if c_total > 0 else 0.0

        # Build feature dictionary
        row = {
            "sequence_id": record.id,
            "filename": filename,
            "source": "NCBI GenBank",
            "organism": organism,
            "taxonomy": "Bacteria",
            "sequence_type": seq_type,
            "length_bp": length,
            "gc_content_pct": round(gc_pct, 2),
            "at_content_pct": round(at_pct, 2),
            "n_content_pct": round(n_pct, 2),
            "a_frequency": round(a_freq, 4),
            "t_frequency": round(t_freq, 4),
            "c_frequency": round(c_freq, 4),
            "g_frequency": round(g_freq, 4),
            "gc_skew": round(gc_skew, 4),
            "at_skew": round(at_skew, 4),
            "sequence_entropy": calculate_entropy(seq),
            "homopolymer_max_length": homo_max,
            "homopolymer_mean_length": round(homo_mean, 2),
            "homopolymer_count": homo_count,
            "homopolymer_A_count": homo_a,
            "homopolymer_T_count": homo_t,
            "homopolymer_C_count": homo_c,
            "homopolymer_G_count": homo_g,
            "repeat_count": repeat_count,
            "repeat_density": round(repeat_density, 4),
            "low_complexity_fraction": round(low_complexity_fraction, 4),
            "orf_count": orf_count,
            "coding_region_count": orf_count,
            "coding_fraction": round(coding_fraction, 4),
            "average_orf_length": round(avg_orf_len, 2),
            "max_orf_length": max_orf_len,
            "codon_adaptation_index": round(coding_fraction * (gc3_pct / 100), 4),
            "codon_bias_score": round(codon_bias, 6),
            "gc1_pct": round(gc1_pct, 2),
            "gc2_pct": round(gc2_pct, 2),
            "gc3_pct": round(gc3_pct, 2),
            "is_synthetic": is_synthetic
        }

        row.update(k2_counts)
        row.update(k3_counts)
        row.update(k4_counts)
        row.update(codon_counts)

        records.append(row)

    return records

if __name__ == "__main__":
    # Explicit raw string path to prevent backslash/unicode escape issues
    natural_dir = r"C:\Users\vinay\OneDrive\Desktop\VMedithon\demo_samples\Natural DNAs"
    natural_records = []
    
    if os.path.exists(natural_dir):
        # Sort files sequentially from natural_1.fasta through natural_20.fasta
        files = sorted(
            [f for f in os.listdir(natural_dir) if f.startswith("natural_") and f.endswith(('.fasta', '.fa', '.txt'))],
            key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
        )
        
        for file in files:
            filepath = os.path.join(natural_dir, file)
            print(f"Extracting [Natural]: {file}")
            natural_records.extend(process_fasta_file(filepath, file))

        # Save Natural CSV inside the Natural DNAs directory
        df_natural = pd.DataFrame(natural_records)
        output_csv = os.path.join(natural_dir, "natural_features.csv")
        df_natural.to_csv(output_csv, index=False)
        
        print(f"\nSUCCESS! Generated {output_csv}")
        print(f"Total Natural files processed: {len(df_natural)}")
        print(f"Total features extracted per file: {df_natural.shape[1]}")
    else:
        print(f"Directory non-existent: '{natural_dir}'. Please verify the path.")