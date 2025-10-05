#!/usr/bin/env python3

import pandas as pd
import numpy as np
from collections import Counter

# List of your trusted prediction files
prediction_files = [
    'ULTIMATE_95PLUS_SUBMISSION.csv',  # Model A predictions
    'FINAL_CHAMPION_99PLUS.csv',       # Model B predictions
    'ULTIMATE_SUPER_ENSEMBLE.csv'      # Model C predictions
]

# Load all prediction DataFrames
data_frames = []
for file_path in prediction_files:
    try:
        df = pd.read_csv(file_path, header=None, names=['ID', 'TARGET'])
        data_frames.append(df)
        print(f"Successfully loaded {file_path}")
    except FileNotFoundError:
        print(f"Warning: {file_path} not found, skipping")

# Proceed only if at least two models are available
if len(data_frames) < 2:
    print("Not enough models to ensemble. Exiting.")
    exit(1)

# Merge all predictions on 'ID'
ensemble_base = data_frames[0].copy()
for idx, df in enumerate(data_frames[1:], start=1):
    ensemble_base = ensemble_base.merge(
        df.rename(columns={'TARGET': f'model_{idx}'}),
        on='ID'
    )

# Strategy 1: Simple majority vote among all models
def simple_majority(row):
    votes = [row['TARGET']] + [row[f'model_{i}'] for i in range(1, len(data_frames))]
    return Counter(votes).most_common(1)[0][0]

ensemble_simple = ensemble_base.copy()
ensemble_simple['TARGET'] = ensemble_simple.apply(simple_majority, axis=1)
ensemble_simple[['ID', 'TARGET']].to_csv('ensemble_simple.csv', index=False)
print("Saved: ensemble_simple.csv")

# Strategy 2: Weighted vote favoring Model A (first file)
def weighted_majority(row):
    # Assign triple weight to the first model
    weights = [row['TARGET']] * 3
    other_votes = [row[f'model_{i}'] for i in range(1, len(data_frames))]
    votes = weights + other_votes
    return Counter(votes).most_common(1)[0][0]

ensemble_weighted = ensemble_base.copy()
ensemble_weighted['TARGET'] = ensemble_weighted.apply(weighted_majority, axis=1)
ensemble_weighted[['ID', 'TARGET']].to_csv('ensemble_weighted.csv', index=False)
print("Saved: ensemble_weighted.csv")

# Strategy 3: Conservative consensus (change only if all secondary models agree)
def conservative_consensus(row):
    primary_vote = row['TARGET']
    others = [row[f'model_{i}'] for i in range(1, len(data_frames))]
    if len(set(others)) == 1 and others[0] != primary_vote:
        return others[0]
    return primary_vote

ensemble_conservative = ensemble_base.copy()
ensemble_conservative['TARGET'] = ensemble_conservative.apply(conservative_consensus, axis=1)
ensemble_conservative[['ID', 'TARGET']].to_csv('ensemble_conservative.csv', index=False)
print("Saved: ensemble_conservative.csv")

print("All ensemble files generated. Try submitting to see which performs best!")
