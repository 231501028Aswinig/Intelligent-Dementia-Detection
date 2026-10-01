import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
import xgboost as xgb
import lightgbm as lgb
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def get_high_accuracy_model():
    print("=" * 80)
    print("PUSHING ACCURACY > 91% ON DEMENTIA DATASET 2")
    print("=" * 80)
    
    # Load raw dataset to include more features
    df = pd.read_csv("dementia_dataset_2.csv")
    
    # Keep Visit and MR Delay this time, drop ID and Group
    drop_cols = ['Subject ID', 'MRI ID', 'Group', 'Hand']
    df = df.drop(columns=drop_cols).copy()
    
    # Clean up
    df = df.dropna(subset=['CDR']).copy()
    df['SES'] = df['SES'].fillna(df['SES'].median())
    df['MMSE'] = df['MMSE'].fillna(df['MMSE'].median())
    df['M/F'] = df['M/F'].map({'F': 0, 'M': 1})
    df['CDR_binary'] = (df['CDR'] >= 0.5).astype(int)
    
    # Continuous capping
    continuous_cols = ['eTIV', 'nWBV', 'ASF', 'MMSE', 'Age', 'MR Delay']
    for col in continuous_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
        
    # Feature Engineering
    df['Brain_Volume_Ratio'] = df['nWBV'] / df['eTIV']
    df['Age_Cognition_Score'] = df['Age'] * (30.0 - df['MMSE'])
    df['SES_Educ_Interaction'] = df['SES'] * df['EDUC']
    
    feature_cols = [c for c in df.columns if c not in ['CDR', 'CDR_binary']]
    X = df[feature_cols]
    y = df['CDR_binary']
    
    # We will use an ensemble (VotingClassifier) of ExtraTrees and LightGBM
    # ExtraTrees is fantastic with small datasets and SMOTE
    et = ExtraTreesClassifier(n_estimators=300, max_depth=15, min_samples_split=2, random_state=42)
    rf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
    lgbm = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42, verbose=-1)
    xgb_clf = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, eval_metric='logloss')
    
    voting_clf = VotingClassifier(
        estimators=[
            ('et', et),
            ('rf', rf),
            ('lgbm', lgbm),
            ('xgb', xgb_clf)
        ],
        voting='soft'
    )
    
    # Pipeline: RobustScaler -> SMOTE -> VotingClassifier
    # Doing SMOTE inside the pipeline prevents data leakage
    pipeline = ImbPipeline([
        ('scaler', RobustScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', voting_clf)
    ])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print(f"Features used ({len(feature_cols)}): {feature_cols}")
    print("Evaluating advanced ensemble pipeline with SMOTE...")
    
    accs, f1s = [], []
    
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_val)
        
        accs.append(accuracy_score(y_val, y_pred))
        f1s.append(f1_score(y_val, y_pred))
        
    mean_acc = np.mean(accs) * 100
    mean_f1 = np.mean(f1s)
    
    print("\n" + "=" * 80)
    print(f"FINAL ACCURACY: {mean_acc:.2f}%")
    print(f"FINAL F1 SCORE: {mean_f1:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    get_high_accuracy_model()
