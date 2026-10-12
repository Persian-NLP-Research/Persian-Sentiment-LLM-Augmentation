"""
Text Dataset Intrinsic Quality and Diversity Auditor.

Computes Exact Duplicate Rate, Pairwise Jaccard Near-Duplicate Rates, Mean
Jaccard Distance, Type-Token Ratio (TTR / Distinct-1), and Distinct-2.
"""

import argparse
import itertools
import os
import re
import sys
import pandas as pd
from typing import Union

def clean_text(text: str) -> str:
    #Basic whitespace normalization.
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def run_quality_audit(file_path: str, text_col: Union[str, int] = 0) -> dict:
    #Runs intrinsic quality metrics on the given dataset file.
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # 1. Load data
    try:
        # First attempt: assume no header
        df = pd.read_csv(file_path, header=None)
        if isinstance(text_col, int):
            texts = df.iloc[:, text_col].astype(str)
        else:
            df = pd.read_csv(file_path)
            texts = df[text_col].astype(str)
    except Exception as e:
        # Fallback reading
        df = pd.read_csv(file_path)
        col = text_col if isinstance(text_col, str) else df.columns[text_col]
        texts = df[col].astype(str)

    cleaned_texts = [clean_text(t) for t in texts if clean_text(t)]
    total_samples = len(cleaned_texts)

    if total_samples < 2:
        raise ValueError(
            "Dataset must contain at least 2 samples for pairwise evaluation."
        )

    # 2. Exact Duplicates
    unique_texts = set(cleaned_texts)
    unique_samples = len(unique_texts)
    exact_duplicates = total_samples - unique_samples
    exact_duplicate_rate = (exact_duplicates / total_samples) * 100.0

    # 3. Near-Duplicates via Pairwise Jaccard
    tokens_per_sentence = [set(s.split()) for s in cleaned_texts]
    total_pairs = (total_samples * (total_samples - 1)) // 2

    jaccard_above_80 = 0
    jaccard_above_70 = 0
    total_jaccard_sim = 0.0

    for s1, s2 in itertools.combinations(tokens_per_sentence, 2):
        inter = len(s1.intersection(s2))
        union = len(s1.union(s2))
        sim = inter / union if union > 0 else 0.0
        total_jaccard_sim += sim

        # Exclude exact duplicates (sim == 1.0)
        if 0.80 < sim < 1.0:
            jaccard_above_80 += 1
        if 0.70 < sim < 1.0:
            jaccard_above_70 += 1

    mean_jaccard_sim = total_jaccard_sim / total_pairs
    mean_jaccard_dist = 1.0 - mean_jaccard_sim
    rate_above_80 = (jaccard_above_80 / total_pairs) * 100.0
    rate_above_70 = (jaccard_above_70 / total_pairs) * 100.0

    # 4. Lexical Diversity (TTR / Distinct-1 & Distinct-2)
    all_words = " ".join(cleaned_texts).split()
    total_tokens = len(all_words)
    unique_words = set(all_words)
    ttr = (len(unique_words) / total_tokens) if total_tokens > 0 else 0.0

    if total_tokens > 1:
        bigrams = [
            (all_words[i], all_words[i + 1]) for i in range(total_tokens - 1)
        ]
        distinct_2 = len(set(bigrams)) / len(bigrams)
    else:
        distinct_2 = 0.0

    results = {
        "Total Samples": total_samples,
        "Unique Samples": unique_samples,
        "Exact Duplicates": exact_duplicates,
        "Exact Duplicate Rate (%)": round(exact_duplicate_rate, 2),
        "Total Pairs Evaluated": total_pairs,
        "Near-Duplicates (>0.80)": jaccard_above_80,
        "Near-Duplicate Rate (>0.80) (%)": round(rate_above_80, 6),
        "Near-Duplicates (>0.70)": jaccard_above_70,
        "Near-Duplicate Rate (>0.70) (%)": round(rate_above_70, 6),
        "Mean Pairwise Jaccard Distance": round(mean_jaccard_dist, 4),
        "Total Word Tokens": total_tokens,
        "Unique Word Types": len(unique_words),
        "TTR (Distinct-1)": round(ttr, 4),
        "Distinct-2": round(distinct_2, 4),
    }

    # 5. Print Formatted Report
    print("=" * 65)
    print("        TEXT DATASET INTRINSIC QUALITY AUDIT REPORT        ")
    print("=" * 65)
    for metric, val in results.items():
        if isinstance(val, float):
            print(f" {metric:<35} : {val:.4f}" if val < 1 else f" {metric:<35} : {val}")
        else:
            print(f" {metric:<35} : {val:,}")
    print("=" * 65)

    return results

if __name__ == "__main__":
    
    try:
        import google.colab  # noqa: F401
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    parser = argparse.ArgumentParser(
        description="Audit the quality and diversity of a text dataset."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a CSV file containing the text data.",
    )
    parser.add_argument(
        "--text_col",
        default=0,
        help="Column index (e.g. 0) or column name containing the text.",
    )

    args, unknown = parser.parse_known_args()
    selected_file = args.file

    if not selected_file and IN_COLAB:
        print("Please upload your CSV file:")
        from google.colab import files
        uploaded = files.upload()
        if uploaded:
            selected_file = list(uploaded.keys())[0]

    if not selected_file:
        selected_file = input("Enter path to your CSV file: ").strip()

    if not selected_file:
        print("Error: no file provided.")
        sys.exit(1)

    run_quality_audit(selected_file, text_col=args.text_col)
