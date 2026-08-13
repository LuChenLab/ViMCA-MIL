#!/usr/bin/env python3
import argparse
import gzip
import re
import csv
from collections import defaultdict

import pandas as pd
from Bio.Data import CodonTable


DNA_TABLE = CodonTable.unambiguous_dna_by_name["Standard"]
VALID_BASES = set("ACGT")


def open_maybe_gz(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def fasta_iter(path):
    header = None
    seq_chunks = []

    with open_maybe_gz(path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks).upper().replace("U", "T")
                header = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)

        if header is not None:
            yield header, "".join(seq_chunks).upper().replace("U", "T")


def extract_seq_id(header):
    m = re.search(r"EPI_ISL_\d+", header)
    if m:
        return m.group(0)
    return header.split()[0].split("|")[0]


def translate_codon(codon):
    if len(codon) != 3:
        return None
    if any(b not in VALID_BASES for b in codon):
        return None
    if codon in DNA_TABLE.stop_codons:
        return "*"
    return DNA_TABLE.forward_table.get(codon)


def classify_nm_sm(ref_aa, alt_aa):
  
    if ref_aa is None or alt_aa is None:
        return None

    if ref_aa == "*" or alt_aa == "*":
        return None

    if ref_aa == alt_aa:
        return "Sm"

    return "Nm"


def build_ref_coordinate_map(ref_aligned):
    """
    alignment column -> reference coordinate.
    Columns that are gaps in the reference are marked as -1.
    """
    aln_to_ref_pos = []
    ref_pos = 0
    ref_ungapped = []

    for base in ref_aligned:
        if base == "-":
            aln_to_ref_pos.append(-1)
        else:
            ref_pos += 1
            aln_to_ref_pos.append(ref_pos)
            ref_ungapped.append(base)

    return aln_to_ref_pos, "".join(ref_ungapped)


def load_features(feature_tsv):
    """
    The feature file must contain the columns:
    feature, start, end, ORF, gene

    A single gene may consist of multiple segments, e.g. nsp12:
    nsp12 13442-13468
    nsp12 13468-16236

    All segments of the same gene are concatenated in file order.
    """
    df = pd.read_csv(feature_tsv, sep="\t")

    required = {"feature", "start", "end", "ORF", "gene"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Feature file missing columns: {missing}")

    features = []

    for gene, sub in df.groupby("gene", sort=False):
        segments = []

        for _, row in sub.iterrows():
            start = int(row["start"])
            end = int(row["end"])

            if end < start:
                raise ValueError(f"Invalid interval for {gene}: {start}-{end}")

            segments.append((start, end))

        features.append({
            "gene": gene,
            "segments": segments
        })

    return features


def build_coding_context(ref_ungapped, features):
    """
    ref_pos_to_contexts[ref_pos] = list of contexts.

    A list is used to allow:
    1. the nsp11 / nsp12 overlap
    2. the nsp12 frameshift (position 13468 is reused in the nsp12 transcript)
    3. the ORF7a / ORF7b overlap
    """
    ref_pos_to_contexts = defaultdict(list)

    for feat in features:
        gene = feat["gene"]
        segments = feat["segments"]

        transcript_positions = []
        transcript_bases = []

        for start, end in segments:
            for pos in range(start, end + 1):
                if pos < 1 or pos > len(ref_ungapped):
                    raise ValueError(
                        f"{gene} coordinate {pos} outside reference length {len(ref_ungapped)}"
                    )

                transcript_positions.append(pos)
                transcript_bases.append(ref_ungapped[pos - 1])

        usable_len = len(transcript_bases) - (len(transcript_bases) % 3)

        if usable_len != len(transcript_bases):
            print(
                f"[WARN] {gene}: coding length {len(transcript_bases)} not divisible by 3; "
                f"last {len(transcript_bases) - usable_len} nt ignored."
            )

        for tx_i in range(usable_len):
            codon_start = (tx_i // 3) * 3
            ref_codon = "".join(transcript_bases[codon_start:codon_start + 3])

            ref_aa = translate_codon(ref_codon)
            codon_pos0 = tx_i % 3
            ref_pos = transcript_positions[tx_i]

            ref_pos_to_contexts[ref_pos].append({
                "gene": gene,
                "codon_pos0": codon_pos0,
                "ref_codon": ref_codon,
                "ref_aa": ref_aa
            })

    return ref_pos_to_contexts


def init_gene_count():
    return {
        "Nm": 0,
        "Sm": 0
    }


def main():
    parser = argparse.ArgumentParser(
        description="Count per-sequence per-gene Nm and Sm directly from SARS-CoV-2 MSA."
    )

    parser.add_argument(
        "--msa",
        required=True,
        help="Input MSA FASTA or FASTA.GZ. First record should be reference."
    )
    parser.add_argument(
        "--features",
        required=True,
        help="TSV annotation file with columns: feature, start, end, ORF, gene."
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output TSV: seq_id, gene, Nm, Sm."
    )
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Skip first FASTA record."
    )
    parser.add_argument(
        "--max-seqs",
        type=int,
        default=None,
        help="Optional limit for testing."
    )
    parser.add_argument(
        "--min-valid-bases",
        type=int,
        default=29000,
        help="Skip sequence if valid A/C/G/T bases on reference columns below this."
    )
    parser.add_argument(
        "--include-zero-genes",
        action="store_true",
        help="Write all genes for every sequence, including Nm=0 and Sm=0."
    )
    parser.add_argument(
        "--exclude-genes",
        default="",
        help="Comma-separated genes to exclude, e.g. nsp11."
    )

    args = parser.parse_args()

    exclude_genes = set(x for x in args.exclude_genes.split(",") if x)

    fasta = fasta_iter(args.msa)

    ref_header, ref_aligned = next(fasta)
    ref_aligned = ref_aligned.upper().replace("U", "T")

    aln_to_ref_pos, ref_ungapped = build_ref_coordinate_map(ref_aligned)

    print(f"[INFO] Reference: {ref_header}")
    print(f"[INFO] Alignment length: {len(ref_aligned)}")
    print(f"[INFO] Ungapped reference length: {len(ref_ungapped)}")

    features = load_features(args.features)
    gene_names = [f["gene"] for f in features if f["gene"] not in exclude_genes]

    ref_pos_to_contexts = build_coding_context(ref_ungapped, features)

    print(f"[INFO] Gene/NSP groups: {len(features)}")
    print(f"[INFO] Annotated reference positions: {len(ref_pos_to_contexts)}")
    print(f"[INFO] Excluded genes: {','.join(sorted(exclude_genes)) if exclude_genes else 'None'}")

    output_fields = ["seq_id", "gene", "Nm", "Sm"]

    n_seq = 0
    n_out_rows = 0

    with open(args.out, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=output_fields, delimiter="\t")
        writer.writeheader()

        if args.skip_reference:
            seq_iter = fasta
        else:
            seq_iter = [(ref_header, ref_aligned)]
            seq_iter.extend(fasta)

        for header, seq_aligned in seq_iter:
            n_seq += 1

            if args.max_seqs is not None and n_seq > args.max_seqs:
                break

            seq_id = extract_seq_id(header)

            if len(seq_aligned) != len(ref_aligned):
                print(
                    f"[WARN] Skip {seq_id}: alignment length mismatch "
                    f"{len(seq_aligned)} != {len(ref_aligned)}"
                )
                continue

            valid_bases = 0
            for aln_i, q_base in enumerate(seq_aligned):
                ref_pos = aln_to_ref_pos[aln_i]
                if ref_pos == -1:
                    continue
                if q_base in VALID_BASES:
                    valid_bases += 1

            if valid_bases < args.min_valid_bases:
                continue

            seq_counts = defaultdict(init_gene_count)

            for aln_i, (r_base, q_base) in enumerate(zip(ref_aligned, seq_aligned)):
                ref_pos = aln_to_ref_pos[aln_i]

                if ref_pos == -1:
                    continue

                if r_base not in VALID_BASES:
                    continue
                if q_base not in VALID_BASES:
                    continue
                if q_base == r_base:
                    continue

                contexts = ref_pos_to_contexts.get(ref_pos)
                if not contexts:
                    continue

                for ctx in contexts:
                    gene = ctx["gene"]

                    if gene in exclude_genes:
                        continue

                    ref_codon = ctx["ref_codon"]
                    codon_pos0 = ctx["codon_pos0"]

                    alt_codon_list = list(ref_codon)
                    alt_codon_list[codon_pos0] = q_base
                    alt_codon = "".join(alt_codon_list)

                    ref_aa = ctx["ref_aa"]
                    alt_aa = translate_codon(alt_codon)

                    cls = classify_nm_sm(ref_aa, alt_aa)

                    if cls == "Nm":
                        seq_counts[gene]["Nm"] += 1
                    elif cls == "Sm":
                        seq_counts[gene]["Sm"] += 1
                    else:
                        # Codons that cannot be classified are not counted as Nm/Sm
                        pass

            if args.include_zero_genes:
                genes_to_write = gene_names
            else:
                genes_to_write = sorted(seq_counts.keys())

            for gene in genes_to_write:
                if gene in exclude_genes:
                    continue

                Nm = seq_counts[gene]["Nm"]
                Sm = seq_counts[gene]["Sm"]

                if not args.include_zero_genes and Nm == 0 and Sm == 0:
                    continue

                writer.writerow({
                    "seq_id": seq_id,
                    "gene": gene,
                    "Nm": Nm,
                    "Sm": Sm
                })
                n_out_rows += 1

            if n_seq % 10000 == 0:
                print(f"[INFO] Processed sequences: {n_seq}; output rows: {n_out_rows}")

    print(f"[DONE] Processed sequences: {n_seq}")
    print(f"[DONE] Output rows: {n_out_rows}")
    print(f"[DONE] Output file: {args.out}")


if __name__ == "__main__":
    main()
