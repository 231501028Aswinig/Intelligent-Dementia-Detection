import os
import pandas as pd
import numpy as np
from compare_models import evaluate_cleaned_dataset

def prepare_dataset_2(file_path):
    df_raw = pd.read_csv(file_path)
    
    # 1. Drop metadata and leak columns
    drop_cols = ['Subject ID', 'MRI ID', 'Group', 'Visit', 'MR Delay', 'Hand']
    df = df_raw.drop(columns=drop_cols).copy()
    
    # 2. Rename EDUC to Educ to match the old dataset
    df = df.rename(columns={'EDUC': 'Educ'})
    
    # 3. Filter CDR non-null rows (just in case, though it has 0 missing)
    df = df.dropna(subset=['CDR']).copy()
    
    # 4. Impute missing values (SES and MMSE)
    df['SES'] = df['SES'].fillna(df['SES'].median())
    df['MMSE'] = df['MMSE'].fillna(df['MMSE'].median())
    
    # 5. Encode 'M/F' as binary (0/1)
    df['M/F'] = df['M/F'].map({'F': 0, 'M': 1})
    
    # 6. Create binary target 'CDR_binary'
    df['CDR_binary'] = (df['CDR'] >= 0.5).astype(int)
    
    # 7. IQR Outlier Capping on continuous features
    continuous_cols = ['eTIV', 'nWBV', 'ASF', 'MMSE', 'Age']
    for col in continuous_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
    # 8. Feature Engineering
    df['Brain_Volume_Ratio'] = df['nWBV'] / df['eTIV']
    df['Age_Cognition_Score'] = df['Age'] * (30.0 - df['MMSE'])

    output_path = os.path.join("data", "cleaned_dataset_2.csv")
    df.to_csv(output_path, index=False)
    return output_path

def main():
    cleaned_oasis_path = os.path.join("data", "cleaned_oasis.csv")
    dataset_2_raw_path = "dementia_dataset_2.csv"
    
    # Prepare second dataset
    cleaned_dataset_2_path = prepare_dataset_2(dataset_2_raw_path)
    
    print("Evaluating Original Dataset (cleaned_oasis.csv)...")
    df_results_1, _ = evaluate_cleaned_dataset(cleaned_oasis_path)
    
    print("\nEvaluating Second Dataset (cleaned_dataset_2.csv)...")
    df_results_2, _ = evaluate_cleaned_dataset(cleaned_dataset_2_path)
    
    print("\n" + "=" * 80)
    print("DATASET 1: Cross-Sectional (cleaned_oasis.csv) - 5-Fold CV")
    print("=" * 80)
    print(df_results_1.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("DATASET 2: Longitudinal (cleaned_dataset_2.csv) - 5-Fold CV")
    print("=" * 80)
    print(df_results_2.to_string(index=False))
    
    # Save the output to a text file for the user to see easily
    with open("dataset_comparison_results.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("DATASET 1: Cross-Sectional (cleaned_oasis.csv) - 5-Fold CV\n")
        f.write("=" * 80 + "\n")
        f.write(df_results_1.to_string(index=False) + "\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("DATASET 2: Longitudinal (cleaned_dataset_2.csv) - 5-Fold CV\n")
        f.write("=" * 80 + "\n")
        f.write(df_results_2.to_string(index=False) + "\n")
        
if __name__ == "__main__":
    main()
