import os
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction

# Get the path of the current folder
folder_path = os.path.dirname(os.path.abspath(__file__))

records = []

for filename in os.listdir(folder_path):
    # Process files starting with synthetic or natural
    if filename.lower().startswith(("synthetic", "natural")):
        filepath = os.path.join(folder_path, filename)
        
        # Determine target label based on filename
        label = 1 if "synthetic" in filename.lower() else 0
        
        try:
            for record in SeqIO.parse(filepath, "fasta"):
                seq_str = str(record.seq).upper()
                records.append({
                    "filename": filename,
                    "length": len(seq_str),
                    "gc_content": round(gc_fraction(seq_str) * 100, 2),
                    "is_synthetic": label
                })
        except Exception as e:
            print(f"Could not parse {filename}: {e}")

# Save output summary
if records:
    df = pd.DataFrame(records)
    output_csv = os.path.join(folder_path, "dataset_summary.csv")
    df.to_csv(output_csv, index=False)
    print("SUCCESS! Parsed sequences:")
    print(df)
else:
    print("No matching synthetic_* or natural_* files found in this folder.")