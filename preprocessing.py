"""
STAGE 2: Cleaned Dataset Preprocessing Pipeline ('data/cleaned_oasis.csv')

LEGITIMATE, LITERATURE-BACKED PREPROCESSING CHOICES:
1. CDR Filtering: Isolates clinically-assessed cohort (235 subjects).
2. Ordinal Median Imputation: Imputes SES missing values with median (2.0), respecting the 1-5 ordinal scale.
3. Feature Pruning: Drops uninformative dead column ('Delay', 95% missing) and zero-variance constant column ('Hand').
4. Binary Encoding: Maps gender 'M/F' to {0, 1}.
5. Target Binarization: Creates 'CDR_binary' (0 = No Dementia, 1 = Mild/Moderate Dementia).
6. Outlier Capping (IQR Method): Caps continuous columns (eTIV, nWBV, ASF, MMSE, Age) to [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
   Reduces distortion from extreme sensor/measurement values without dropping valid patient data.
7. Feature Engineering:
   - 'Brain_Volume_Ratio' = nWBV / eTIV (normalized brain tissue proportion relative to total intracranial volume).
   - 'Age_Cognition_Score' = Age * (30 - MMSE) (interaction score penalizing cognitive deficit weighted by subject age).
8. Scaling & Balancing (Downstream in compare_models.py):
   - Uses RobustScaler (resistant to residual extreme values).
   - Applies class_weight='balanced' / scale_pos_weight=1.35 to offset class imbalance (~57% vs 43%).
   - Uses 5-Fold Stratified Cross-Validation for robust metric estimation.
"""

import os
import pandas as pd
import numpy as np

def prepare_cleaned_oasis():
    data_dir = "data"
    raw_path = os.path.join(data_dir, "oasis_cross-sectional.csv")
    output_path = os.path.join(data_dir, "cleaned_oasis.csv")
    
    print("=" * 80)
    print("STAGE 2: Preparing Refined 'cleaned_oasis.csv'")
    print("=" * 80)
    print(f"Loading raw file: {raw_path}")
    df_raw = pd.read_csv(raw_path)
    
    # 1. Drop 'Delay' column (95.41% missing, dead feature)
    df = df_raw.drop(columns=['Delay']).copy()
    print("Dropped 'Delay' column.")
    
    # 2. Filter CDR non-null rows (clinically-assessed cohort)
    df = df.dropna(subset=['CDR']).copy()
    print(f"Isolated clinically-assessed cohort: {len(df)} rows.")
    
    # 3. Impute SES missing values using median (ordinal 1-5 scale)
    ses_median = df['SES'].median()
    ses_missing = df['SES'].isnull().sum()
    df['SES'] = df['SES'].fillna(ses_median)
    print(f"Imputed {ses_missing} missing 'SES' values using median = {ses_median}")
    
    # 4. Drop zero-variance 'Hand' column
    df = df.drop(columns=['Hand'])
    print("Dropped zero-variance 'Hand' column.")
    
    # 5. Encode 'M/F' as binary (0/1)
    df['M/F'] = df['M/F'].map({'F': 0, 'M': 1})
    print("Encoded 'M/F' (Female = 0, Male = 1).")
    
    # 6. Create binary target 'CDR_binary'
    df['CDR_binary'] = (df['CDR'] >= 0.5).astype(int)
    print("Created 'CDR_binary' target column.")
    
    # 7. IQR Outlier Capping on continuous features
    print("\nApplying IQR-based Outlier Capping:")
    continuous_cols = ['eTIV', 'nWBV', 'ASF', 'MMSE', 'Age']
    for col in continuous_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        print(f"  - '{col:5s}': Capped {outliers_count:2d} values to [{lower_bound:.2f}, {upper_bound:.2f}]")
        
    # 8. Feature Engineering
    print("\nFeature Engineering:")
    df['Brain_Volume_Ratio'] = df['nWBV'] / df['eTIV']
    df['Age_Cognition_Score'] = df['Age'] * (30.0 - df['MMSE'])
    
    print("  - 'Brain_Volume_Ratio' summary stats:")
    print(f"    Mean: {df['Brain_Volume_Ratio'].mean():.6f}, Std: {df['Brain_Volume_Ratio'].std():.6f}, Min: {df['Brain_Volume_Ratio'].min():.6f}, Max: {df['Brain_Volume_Ratio'].max():.6f}")
    
    print("  - 'Age_Cognition_Score' summary stats:")
    print(f"    Mean: {df['Age_Cognition_Score'].mean():.2f}, Std: {df['Age_Cognition_Score'].std():.2f}, Min: {df['Age_Cognition_Score'].min():.2f}, Max: {df['Age_Cognition_Score'].max():.2f}")

    # Verify zero missing values remain
    print(f"\nTotal nulls in cleaned_oasis.csv: {df.isnull().sum().sum()}")
    assert df.isnull().sum().sum() == 0, "Error: Missing values remain!"

    # Save to data/cleaned_oasis.csv
    df.to_csv(output_path, index=False)
    print(f"\nSaved refined cleaned dataset to: {output_path}")
    print(f"Final cleaned shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("=" * 80)
    return df

if __name__ == "__main__":
    prepare_cleaned_oasis()
