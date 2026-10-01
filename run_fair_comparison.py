import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def preprocess_and_evaluate(df, dataset_name):
    # 1. Standard Preprocessing
    df = df.dropna(subset=['CDR']).copy()
    
    if 'SES' in df.columns:
        df['SES'] = df['SES'].fillna(df['SES'].median())
    if 'MMSE' in df.columns:
        df['MMSE'] = df['MMSE'].fillna(df['MMSE'].median())
        
    if 'M/F' in df.columns:
        df['M/F'] = df['M/F'].map({'F': 0, 'M': 1, 0: 0, 1: 1})
        
    df['CDR_binary'] = (df['CDR'] >= 0.5).astype(int)
    
    # 2. Feature Engineering
    df['Brain_Volume_Ratio'] = df['nWBV'] / df['eTIV']
    df['Age_Cognition_Score'] = df['Age'] * (30.0 - df['MMSE'])
    
    # 3. Drop leak and identifier columns
    # We use a comprehensive list to catch columns from both datasets
    drop_candidates = ['ID', 'Subject ID', 'MRI ID', 'Group', 'Hand', 'CDR', 'CDR_binary', 'Delay', 'MR Delay', 'Visit']
    drop_cols = [c for c in drop_candidates if c in df.columns]
    
    # Also standardize 'Educ' vs 'EDUC'
    if 'EDUC' in df.columns:
        df = df.rename(columns={'EDUC': 'Educ'})
        
    X = df.drop(columns=drop_cols)
    y = df['CDR_binary']
    
    # 4. Train/Test Split (Exact same seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=22, stratify=y
    )
    
    # 5. SMOTE Optimization (Exact same configuration)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # 6. ExtraTrees Model (Exact same configuration)
    model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    
    # 7. Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"{dataset_name} Final Accuracy: {acc * 100:.2f}%")
    return acc

def main():
    print("=" * 80)
    print("FAIR COMPARISON: EXTRA TREES + SMOTE ON BOTH DATASETS")
    print("=" * 80)
    
    # Dataset 1 (Original Cross-Sectional)
    df1 = pd.read_csv("data/oasis_cross-sectional.csv")
    acc1 = preprocess_and_evaluate(df1, "Dataset 1 (Original Cross-Sectional)")
    
    # Dataset 2 (Longitudinal)
    df2 = pd.read_csv("dementia_dataset_2.csv")
    acc2 = preprocess_and_evaluate(df2, "Dataset 2 (New Longitudinal)")
    
    print("\n" + "=" * 80)
    if acc2 > acc1:
        print(f"CONCLUSION: Dataset 2 is superior by {(acc2 - acc1) * 100:.2f}% under identical optimized conditions.")
    elif acc1 > acc2:
        print(f"CONCLUSION: Dataset 1 is superior by {(acc1 - acc2) * 100:.2f}% under identical optimized conditions.")
    else:
        print("CONCLUSION: Both datasets perform identically under optimized conditions.")
    print("=" * 80)

if __name__ == "__main__":
    main()
