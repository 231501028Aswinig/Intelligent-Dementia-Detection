"""
OASIS Preprocessing Verification Script

Runs automated checks against data/cleaned_oasis.csv, data/oasis_cross-sectional.csv,
and outputs/ to verify dataset integrity, feature encoding, imputation, row counts, and plot outputs.
"""

import os
import pandas as pd

def main():
    data_dir = "data"
    output_dir = "outputs"
    cleaned_path = os.path.join(data_dir, "cleaned_oasis.csv")
    raw_path = os.path.join(data_dir, "oasis_cross-sectional.csv")
    
    results = {}
    
    print("=" * 80)
    print("OASIS PREPROCESSING PIPELINE VERIFICATION")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # CHECK 1: FILE EXISTS & LOADS
    # -------------------------------------------------------------------------
    print("\n--- CHECK 1: FILE EXISTS ---")
    if os.path.exists(cleaned_path):
        try:
            df_cleaned = pd.read_csv(cleaned_path)
            print(f"PASS: '{cleaned_path}' exists and loaded successfully.")
            results['1. FILE EXISTS'] = 'PASS'
        except Exception as e:
            print(f"FAIL: Error loading '{cleaned_path}': {e}")
            results['1. FILE EXISTS'] = 'FAIL'
            return
    else:
        print(f"FAIL: File '{cleaned_path}' does not exist.")
        results['1. FILE EXISTS'] = 'FAIL'
        return

    # -------------------------------------------------------------------------
    # CHECK 2: ROW COUNT
    # -------------------------------------------------------------------------
    print("\n--- CHECK 2: ROW COUNT ---")
    cleaned_rows = len(df_cleaned)
    print(f"Cleaned Dataframe Row Count: {cleaned_rows}")
    if 230 <= cleaned_rows <= 240:
        print(f"PASS: Row count ({cleaned_rows}) is within expected ~235 range.")
        results['2. ROW COUNT'] = 'PASS'
    else:
        print(f"FAIL: Row count ({cleaned_rows}) deviates significantly from expected ~235.")
        results['2. ROW COUNT'] = 'FAIL'

    # -------------------------------------------------------------------------
    # CHECK 3: COLUMN CHECK & DTYPES
    # -------------------------------------------------------------------------
    print("\n--- CHECK 3: COLUMN CHECK ---")
    cols = list(df_cleaned.columns)
    print(f"Final Columns ({len(cols)}): {cols}")
    print("\nColumn Data Types:")
    print(df_cleaned.dtypes)
    
    delay_absent = 'Delay' not in cols
    cdr_binary_present = 'CDR_binary' in cols
    hand_check = 'Hand' not in cols or df_cleaned['Hand'].nunique() > 1
    
    if delay_absent and cdr_binary_present and hand_check:
        print("PASS: 'Delay' is absent, 'CDR_binary' is present, and 'Hand' status is correct.")
        results['3. COLUMN CHECK'] = 'PASS'
    else:
        print(f"FAIL: Column criteria not met. Delay absent: {delay_absent}, CDR_binary present: {cdr_binary_present}, Hand check: {hand_check}")
        results['3. COLUMN CHECK'] = 'FAIL'

    # -------------------------------------------------------------------------
    # CHECK 4: NULL CHECK
    # -------------------------------------------------------------------------
    print("\n--- CHECK 4: NULL CHECK ---")
    null_counts = df_cleaned.isnull().sum()
    print("Null count per column:")
    print(null_counts)
    total_nulls = null_counts.sum()
    print(f"Total nulls in cleaned dataframe: {total_nulls}")
    
    if total_nulls == 0:
        print("PASS: Zero null values exist across all columns.")
        results['4. NULL CHECK'] = 'PASS'
    else:
        print(f"FAIL: Dataframe contains {total_nulls} null values!")
        results['4. NULL CHECK'] = 'FAIL'

    # -------------------------------------------------------------------------
    # CHECK 5: SES IMPUTATION SANITY CHECK
    # -------------------------------------------------------------------------
    print("\n--- CHECK 5: SES IMPUTATION SANITY CHECK ---")
    ses_min = df_cleaned['SES'].min()
    ses_max = df_cleaned['SES'].max()
    ses_median = df_cleaned['SES'].median()
    print(f"SES Statistics in Cleaned Data -> Min: {ses_min}, Max: {ses_max}, Median: {ses_median}")
    
    if 1.0 <= ses_min and ses_max <= 5.0 and 1.0 <= ses_median <= 5.0:
        print("PASS: SES values and median are within the valid 1-5 ordinal scale.")
        results['5. SES IMPUTATION'] = 'PASS'
    else:
        print(f"FAIL: SES values out of range (1-5 scale). Min: {ses_min}, Max: {ses_max}, Median: {ses_median}")
        results['5. SES IMPUTATION'] = 'FAIL'

    # -------------------------------------------------------------------------
    # CHECK 6: TARGET DISTRIBUTION CHECK
    # -------------------------------------------------------------------------
    print("\n--- CHECK 6: TARGET DISTRIBUTION CHECK ---")
    target_counts = df_cleaned['CDR_binary'].value_counts().sort_index()
    print("CDR_binary value counts:")
    print(target_counts)
    total_target_samples = target_counts.sum()
    print(f"Total target samples: {total_target_samples} (matches cleaned rows: {cleaned_rows})")
    
    class_0_cnt = target_counts.get(0, 0)
    class_1_cnt = target_counts.get(1, 0)
    
    if total_target_samples == cleaned_rows and class_0_cnt > 20 and class_1_cnt > 20:
        print(f"PASS: Target distribution is balanced and matches total rows. Class 0: {class_0_cnt}, Class 1: {class_1_cnt}.")
        results['6. TARGET DISTRIBUTION'] = 'PASS'
    else:
        print(f"FAIL: Target distribution issue. Class 0: {class_0_cnt}, Class 1: {class_1_cnt}, Total: {total_target_samples}")
        results['6. TARGET DISTRIBUTION'] = 'FAIL'

    # -------------------------------------------------------------------------
    # CHECK 7: CROSS-CHECK AGAINST ORIGINAL
    # -------------------------------------------------------------------------
    print("\n--- CHECK 7: CROSS-CHECK AGAINST ORIGINAL ---")
    if os.path.exists(raw_path):
        df_raw = pd.read_csv(raw_path)
        raw_filtered = df_raw.dropna(subset=['CDR'])
        raw_filtered_rows = len(raw_filtered)
        print(f"Original dataset CDR non-null row count: {raw_filtered_rows}")
        print(f"Cleaned dataset row count:               {cleaned_rows}")
        
        if raw_filtered_rows == cleaned_rows:
            print("PASS: Cleaned row count EXACTLY matches original filtered row count.")
            results['7. CROSS-CHECK ORIGINAL'] = 'PASS'
        else:
            print(f"FAIL: Row count mismatch! Original filtered: {raw_filtered_rows}, Cleaned: {cleaned_rows}")
            results['7. CROSS-CHECK ORIGINAL'] = 'FAIL'
    else:
        print(f"FAIL: Raw file '{raw_path}' not found.")
        results['7. CROSS-CHECK ORIGINAL'] = 'FAIL'

    # -------------------------------------------------------------------------
    # CHECK 8: M/F ENCODING CHECK
    # -------------------------------------------------------------------------
    print("\n--- CHECK 8: M/F ENCODING CHECK ---")
    unique_gender = set(df_cleaned['M/F'].unique())
    print(f"Unique values in M/F column: {unique_gender}")
    
    if unique_gender == {0, 1}:
        print("PASS: M/F column correctly encoded as binary 0 and 1.")
        results['8. M/F ENCODING'] = 'PASS'
    else:
        print(f"FAIL: M/F unique values {unique_gender} do not match expected {{0, 1}}.")
        results['8. M/F ENCODING'] = 'FAIL'

    # -------------------------------------------------------------------------
    # CHECK 9: PLOT FILES CHECK
    # -------------------------------------------------------------------------
    print("\n--- CHECK 9: PLOT FILES CHECK ---")
    expected_plots = [
        "age_distribution_hist.png",
        "cdr_binary_class_counts.png",
        "mmse_by_cdr_binary.png",
        "nwbv_by_cdr_binary.png",
        "numeric_correlation_heatmap.png"
    ]
    
    output_files = os.listdir(output_dir) if os.path.exists(output_dir) else []
    print(f"Files found in '{output_dir}': {output_files}")
    
    missing_plots = [plot for plot in expected_plots if plot not in output_files]
    
    if len(missing_plots) == 0:
        print("PASS: All 5 required plot files exist in the outputs directory.")
        results['9. PLOT FILES'] = 'PASS'
    else:
        print(f"FAIL: Missing plot files: {missing_plots}")
        results['9. PLOT FILES'] = 'FAIL'

    # -------------------------------------------------------------------------
    # CHECK 10: FINAL SUMMARY TABLE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("10. FINAL SUMMARY TABLE")
    print("=" * 80)
    
    summary_df = pd.DataFrame(list(results.items()), columns=["Check Name", "Status"])
    print(summary_df.to_string(index=False))
    
    all_pass = all(status == 'PASS' for status in results.values())
    print("=" * 80)
    if all_pass:
        print("OVERALL VERIFICATION: ALL CHECKS PASSED SUCCESSFULLY!")
    else:
        print("OVERALL VERIFICATION: SOME CHECKS FAILED - ATTENTION REQUIRED!")
    print("=" * 80)

if __name__ == "__main__":
    main()
