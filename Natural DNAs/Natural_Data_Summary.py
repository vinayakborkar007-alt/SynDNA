import os
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction

# Target directory containing natural_1..5 and synthetic_1..5
folder_path = os.path.dirname(os.path.abspath(__file__))

records = []

for filename in os.listdir(folder_path):
    # Match both natural_* and synthetic_* files
    if filename.lower().startswith(("natural", "synthetic")):
        filepath = os.path.join(folder_path, filename)
        
        # Assign label: 0 for natural genomes, 1 for synthetic vectors
        label = 1 if filename.lower().startswith("synthetic") else 0
        
        try:
            for record in SeqIO.parse(filepath, "fasta"):
                seq_str = str(record.seq).upper()
                records.append({
                    "filename": filename,
                    "sequence_id": record.id,
                    "length_bp": len(seq_str),
                    "gc_content_pct": round(gc_fraction(seq_str) * 100, 2),
                    "is_synthetic": label
                })
        except Exception as e:
            print(f"Skipping {filename}: {e}")

# Save generated summary table
if records:
    df = pd.DataFrame(records)
    output_csv = os.path.join(folder_path, "dataset_summary.csv")
    df.to_csv(output_csv, index=False)
    print("SUCCESS! Processed Dataset Summary:")
    print(df.to_string(index=False))
else:
    print("No 'natural_' or 'synthetic_' files detected in the folder.")