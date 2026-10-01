import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def load_and_preprocess_data(raw_csv_path="dementia_dataset_2.csv", cleaned_csv_path="data/cleaned_dataset_2.csv"):
    """
    Loads and runs the existing preprocessing and feature engineering pipeline on the dataset
    without using any synthetic oversampling (SMOTE) or data leakage.
    
    Returns:
        df_cleaned (pd.DataFrame): Cleaned and feature-engineered dataset.
        feature_cols (list): List of feature column names.
        target_col (str): Name of target column ('CDR_binary').
    """
    # If cleaned dataset already exists, load it; otherwise preprocess from raw
    if os.path.exists(cleaned_csv_path):
        df = pd.read_csv(cleaned_csv_path)
        if 'M/F' in df.columns and df['M/F'].dtype == object:
            df['M/F'] = df['M/F'].astype(str).str.strip().map({'F': 0, 'M': 1, '0': 0, '1': 1, '0.0': 0, '1.0': 1}).fillna(0).astype(int)
    elif os.path.exists(raw_csv_path):
        df_raw = pd.read_csv(raw_csv_path)
        
        # 1. Drop metadata and non-feature columns
        drop_cols = [c for c in ['Subject ID', 'MRI ID', 'Group', 'Visit', 'MR Delay', 'Hand'] if c in df_raw.columns]
        df = df_raw.drop(columns=drop_cols).copy()
        
        # 2. Harmonize column names
        if 'EDUC' in df.columns:
            df = df.rename(columns={'EDUC': 'Educ'})
            
        # 3. Filter CDR non-null rows
        df = df.dropna(subset=['CDR']).copy()
        
        # 4. Impute missing values with medians
        if 'SES' in df.columns:
            df['SES'] = df['SES'].fillna(df['SES'].median())
        if 'MMSE' in df.columns:
            df['MMSE'] = df['MMSE'].fillna(df['MMSE'].median())
            
        # 5. Encode 'M/F' as binary (0 = Female, 1 = Male)
        if 'M/F' in df.columns:
            df['M/F'] = df['M/F'].astype(str).str.strip().map({'F': 0, 'M': 1, '0': 0, '1': 1, '0.0': 0, '1.0': 1, 'False': 0, 'True': 1}).fillna(0).astype(int)
            
        # 6. Binary target definition (0 = Nondemented, 1 = Demented / CDR >= 0.5)
        df['CDR_binary'] = (df['CDR'] >= 0.5).astype(int)
        
        # 7. IQR Outlier Capping on continuous features
        continuous_cols = [c for c in ['eTIV', 'nWBV', 'ASF', 'MMSE', 'Age'] if c in df.columns]
        for col in continuous_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
        # 8. Literature-backed Feature Engineering
        if 'nWBV' in df.columns and 'eTIV' in df.columns:
            df['Brain_Volume_Ratio'] = df['nWBV'] / df['eTIV']
        if 'Age' in df.columns and 'MMSE' in df.columns:
            df['Age_Cognition_Score'] = df['Age'] * (30.0 - df['MMSE'])
            
        # Ensure output directory exists and save
        os.makedirs(os.path.dirname(cleaned_csv_path) or "data", exist_ok=True)
        df.to_csv(cleaned_csv_path, index=False)
    else:
        raise FileNotFoundError(f"Neither {cleaned_csv_path} nor {raw_csv_path} was found.")

    # Define feature columns
    excluded_cols = ['ID', 'Subject ID', 'MRI ID', 'Group', 'Visit', 'MR Delay', 'Hand', 'CDR', 'CDR_binary']
    feature_cols = [c for c in df.columns if c not in excluded_cols]
    target_col = 'CDR_binary'
    
    # Strictly ensure all feature columns are numeric
    for col in feature_cols:
        if col == 'M/F' or df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().map({'F': 0, 'M': 1, '0': 0, '1': 1, '0.0': 0, '1.0': 1, 'False': 0, 'True': 1}).fillna(0)
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(int)
    
    return df, feature_cols, target_col

def get_stratified_split(df, feature_cols, target_col='CDR_binary', test_size=0.20, random_state=42):
    """
    Performs a strict 80:20 stratified train-test split.
    Scales features with RobustScaler fit strictly on training set only.
    Calculates training set class balance weight (scale_pos_weight).
    """
    X = df[feature_cols]
    y = df[target_col]
    
    # Locked 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Fit scaler strictly on training set to prevent data leakage
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)
    
    # Class balance weight from training set
    neg_cnt = (y_train == 0).sum()
    pos_cnt = (y_train == 1).sum()
    scale_pos_weight = neg_cnt / pos_cnt if pos_cnt > 0 else 1.0
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'scaler': scaler,
        'scale_pos_weight': scale_pos_weight,
        'feature_cols': feature_cols
    }
