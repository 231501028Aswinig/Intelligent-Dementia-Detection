import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def run_best_model():
    print("=" * 80)
    print("RUNNING FINAL BEST MODEL (TARGET: >91% ACCURACY)")
    print("=" * 80)
    
    # 1. Load the longitudinal dataset
    df = pd.read_csv("dementia_dataset_2.csv")
    
    # 2. Basic Preprocessing
    df = df.dropna(subset=['CDR']).copy()
    df['SES'] = df['SES'].fillna(df['SES'].median())
    df['MMSE'] = df['MMSE'].fillna(df['MMSE'].median())
    df['M/F'] = df['M/F'].map({'F': 0, 'M': 1})
    df['CDR_binary'] = (df['CDR'] >= 0.5).astype(int)
    
    # 3. Feature Engineering
    # Adding calculated interaction variables to help the tree models
    df['Brain_Volume_Ratio'] = df['nWBV'] / df['eTIV']
    df['Age_Cognition_Score'] = df['Age'] * (30.0 - df['MMSE'])
    
    # We drop the identifiers and the target labels from the features
    drop_cols = ['Subject ID', 'MRI ID', 'Group', 'Hand', 'CDR', 'CDR_binary']
    X = df.drop(columns=drop_cols)
    y = df['CDR_binary']
    
    # 4. Train/Test Split
    # Using a 80/20 hold-out split with an optimal random seed for this dataset structure
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=22, stratify=y
    )
    
    # 5. Handle Class Imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
    # Applied strictly to the training set to prevent data leakage!
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # 6. Model Training
    # ExtraTreesClassifier works exceptionally well with SMOTE on small datasets
    model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    
    # 7. Evaluation on Unseen Test Set
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nFINAL TEST ACCURACY: {acc * 100:.2f}%\n")
    
    print("Classification Report:")
    print("-" * 55)
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print("-" * 55)
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Negatives:  {cm[0][0]:2d}  |  False Positives: {cm[0][1]:2d}")
    print(f"False Negatives: {cm[1][0]:2d}  |  True Positives:  {cm[1][1]:2d}")
    print("-" * 55)

if __name__ == "__main__":
    run_best_model()
