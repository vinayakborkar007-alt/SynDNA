from Bio import SeqIO
from io import StringIO


def parse_fasta(fasta_text: str) -> str:

    fasta_text = fasta_text.strip()

    if not fasta_text:
        raise ValueError("FASTA input is empty")

    records = list(SeqIO.parse(StringIO(fasta_text), "fasta"))

    if not records:
        raise ValueError("Invalid FASTA format")

    sequence = str(records[0].seq).upper()

    valid_bases = set("ATGC")

    if not set(sequence).issubset(valid_bases):
        raise ValueError("Sequence contains invalid DNA characters")

    return sequence