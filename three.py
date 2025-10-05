#!/usr/bin/env python3
"""
blend_submissions.py

Combines all available submission CSVs (including ULTIMATE_SUPER_ENSEMBLE.csv)
using majority voting to maximize accuracy.
"""

import os
import glob
import pandas as pd
from collections import Counter

# Directory containing submission CSVs
BASE = "/home/rithish/Downloads/PALS_ML"

# Find all CSV files that look like submissions
submission_files = [
    f for f in glob.glob(os.path.join(BASE, "*.csv"))
    if os.path.basename(f) not in ("train.csv", "sample_submission.csv")
]

print("🔍 Submission files found:")
for f in submission_files:
    print("  -", os.path.basename(f))

# Read every submission into a DataFrame
dfs = []
for file in submission_files:
    df = pd.read_csv(file)
    # Ensure columns named "ID" and "TARGET"
    df = df.rename(columns={df.columns[0]: "ID", df.columns[1]: "TARGET"})
    dfs.append(df)

# Merge on ID
print("\nMerging submissions on ID...")
merged = dfs[0][["ID"]].copy()
for i, df in enumerate(dfs):
    merged[f"T{i}"] = df["TARGET"]

# Majority voting function
def majority_vote(row):
    votes = [v for v in row if pd.notna(v)]
    return Counter(votes).most_common(1)[0][0]

# Apply majority vote across all columns T0, T1, ...
merged["TARGET"] = merged.drop(columns=["ID"]).apply(majority_vote, axis=1)

# Write final blended submission
output_file = os.path.join(BASE, "blended_submission.csv")
blended = merged[["ID", "TARGET"]]
blended.to_csv(output_file, index=False)

print(f"\n Blended submission saved to {output_file}")
print(f" Contains {len(blended)} predictions by majority vote of {len(dfs)} submissions.")
