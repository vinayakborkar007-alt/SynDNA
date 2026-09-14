# SynDNA

# Data Engineering & Sequence Curation

**Lead:** Vinayak Borkar (Member 1 – Data Lead)[cite: 1]  
**Project:** SynDNA – Automated AI Biosecurity and Synthetic DNA Detection Engine[cite: 1]  
**Event:** VMEDITHON 3.0 (VIT Chennai)[cite: 1]

---

## Overview
As the Data Lead for team **Glitter Girls**, I have curated the initial reference dataset of `.fasta` genomic sequences required to train and validate our XGBoost sequence classification engine[cite: 1]. 

To prepare for feature extraction ($k$-mer profiling, Codon Adaptation Index, GC skew)[cite: 1], I established a balanced ground-truth dataset divided into two primary categories:
* **Class 0 (Natural DNA):** Wild-type bacterial, fungal, and viral genomes downloaded directly from NCBI GenBank[cite: 1].
* **Class 1 (Synthetic / Engineered Vectors):** Standard laboratory cloning vectors and expression plasmids containing synthetic markers, promoters, and cloning backbones[cite: 1].

---

## Curation Summary

### Class 0: Natural Control Group (`label = 0`)
Wild-type reference sequences representing natural evolutionary variation without laboratory modifications[cite: 1]:

* `natural_1.fasta` – *Salmonella enterica* subsp. *enterica* serovar Typhimurium str. LT2 (Complete Genome)
* `natural_2.fasta` – *Staphylococcus aureus* strain MRSA252 (Complete Genome)
* `natural_3.fasta` – *Blastobotrys adeninivorans* LS3 / Yeast (Genomic Chromosome)
* `natural_4.fasta` – *Thermus aquaticus* (Aqualysin Gene Encoding Sequence)
* `natural_5.fasta` – *Enterobacteria phage T4* (Complete Viral Genome)

---

### Class 1: Lab Synthetic & Vector Group (`label = 1`)
Lab-designed plasmids and vectors containing unnaturally clean sequences, cloning scars, and synthetic promoters[cite: 1]:

* `synthetic_1.fasta` – `pUC19` (Cloning Vector)
* `synthetic_2.fasta` – `pBR322` (Complete Sequence Cloning Vector)
* `synthetic_3.fasta` – `pET-28a` (Protein Expression Vector)
* `synthetic_4.fasta` – `pGEM-3Z` (Cloning Vector)
* `synthetic_5.fasta` – `pEGFP-C1` (Enhanced Green Fluorescent Protein Expression Vector)

---

## Current Status & Next Steps

* **[x] Dataset Acquisition:** Downloaded and structured 10 complete `.fasta` sequence files across diverse natural species and synthetic vector types.
* **[x] Label Mapping:** Mapped binary target classes (`0 = Natural`, `1 = Synthetic`) for backend pipeline testing.
* **[x] Demo File Isolation:** Prepared sample files in `/data/demo_samples/` for immediate drag-and-drop testing in the Streamlit UI[cite: 1].
* **[ ] Hackathon Goal:** Support Member 2 (Bioinformatics) and Member 3 (ML Lead) during feature matrix extraction ($k$-mers, CAI, GC skew) and sliding-window regional analysis[cite: 1].
