"""
STAGE 1: Prepare Naive 'raw_minimal.csv' Comparison Dataset (Rebuilt from Scratch)

NAIVE BASELINE PIPELINE (Careless Practitioner Approach):
- Drops rows where CDR target is missing (unavoidable for supervised learning).
- LEAVES all missing values in feature columns (e.g. SES) as NaN — does NOT impute them.
- Keeps low-quality 'Delay' (~95% missing) and zero-variance 'Hand' columns.
- Binary encodes 'M/F' (0/1) and label encodes 'Hand' (0) as minimum necessary for model parsing.
- Binarizes CDR (0 for CDR == 0.0, 1 for CDR >= 0.5).
- Does NOT scale features, does NOT handle outliers, does NOT perform feature engineering.
"""

import os
import pandas as pd
import numpy as np

def prepare_raw_minimal():
    data_dir = "data"
    raw_path = os.path.join(data_dir, "oasis_cross-sectional.csv")
    output_path = os.path.join(data_dir, "raw_minimal.csv")
    
    print("=" * 80)
    print("STAGE 1: Preparing Naive Baseline 'raw_minimal.csv'")
    print("=" * 80)
    print(f"Loading raw file: {raw_path}")
    df_raw = pd.read_csv(raw_path)
    print(f"Initial raw shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    
    # 1. Drop rows where target label CDR is null (necessary for supervised learning)
    df_minimal = df_raw.dropna(subset=['CDR']).copy()
    print(f"Shape after dropping CDR-null rows: {df_minimal.shape[0]} rows, {df_minimal.shape[1]} columns")
    
    # 2. Minimum necessary binary encoding for M/F (0 = Female, 1 = Male)
    gender_map = {'F': 0, 'M': 1}
    df_minimal['M/F'] = df_minimal['M/F'].map(gender_map)
    print("Encoded 'M/F' column as binary (0 = Female, 1 = Male).")

    # Minimum label encoding for Hand ('R' -> 0)
    hand_map = {val: i for i, val in enumerate(df_minimal['Hand'].unique())}
    df_minimal['Hand'] = df_minimal['Hand'].map(hand_map)
    print(f"Encoded 'Hand' column as numeric: {hand_map}")

    # 3. LEAVE NaNs intact in SES and Delay (NO IMPUTATION)
    ses_null_count = df_minimal['SES'].isnull().sum()
    print(f"Leaving {ses_null_count} missing values in 'SES' as NaN (UNIMPUTED).")

    # 4. Create binary target 'CDR_binary'
    df_minimal['CDR_binary'] = (df_minimal['CDR'] >= 0.5).astype(int)
    print("Created 'CDR_binary' target column (0 for CDR == 0.0, 1 for CDR >= 0.5).")
    
    # Print summary of NaNs remaining in dataset
    print(f"\nMissing value summary in raw_minimal.csv:\n{df_minimal.isnull().sum()}")

    # Save to data/raw_minimal.csv
    df_minimal.to_csv(output_path, index=False)
    print(f"\nSaved naive baseline dataset to: {output_path}")
    print(f"Final raw_minimal shape: {df_minimal.shape[0]} rows, {df_minimal.shape[1]} columns")
    print("=" * 80)
    return df_minimal

if __name__ == "__main__":
    prepare_raw_minimal()
