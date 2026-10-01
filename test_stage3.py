import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, confusion_matrix, log_loss
import xgboost as xgb
import lightgbm as lgb

def test_raw():
    df_raw = pd.read_csv("data/raw_minimal.csv")
    drop_cols = ['ID', 'CDR', 'CDR_binary']
    feature_cols = [c for c in df_raw.columns if c not in drop_cols]
    
    X = df_raw[feature_cols]
    y = df_raw['CDR_binary']
    
    # 80/20 Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Raw initial train shape: {X_train.shape}, test shape: {X_test.shape}")
    print(f"Raw train nulls:\n{X_train.isnull().sum()}")

    # For Logistic Regression and Random Forest: drop NaN rows in train/test
    # Note: 'Delay' is 100% NaN. If we drop Delay column for LR/RF or drop NaN rows in SES:
    # Let's drop Delay feature if 100% NaN so row drop affects the 19 SES-NaN rows!
    X_train_no_delay = X_train.drop(columns=['Delay'])
    X_test_no_delay = X_test.drop(columns=['Delay'])
    
    # Drop rows where SES is NaN
    train_valid_mask = X_train_no_delay.notnull().all(axis=1)
    test_valid_mask = X_test_no_delay.notnull().all(axis=1)
    
    X_train_clean = X_train_no_delay[train_valid_mask]
    y_train_clean = y_train[train_valid_mask]
    X_test_clean = X_test_no_delay[test_valid_mask]
    y_test_clean = y_test[test_valid_mask]
    
    print(f"LR/RF train rows dropped due to NaN: {(~train_valid_mask).sum()} (used: {len(X_train_clean)})")
    print(f"LR/RF test rows dropped due to NaN: {(~test_valid_mask).sum()} (used: {len(X_test_clean)})")
    
    # LR
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_clean, y_train_clean)
    y_pred_lr = lr.predict(X_test_clean)
    y_prob_lr = lr.predict_proba(X_test_clean)[:, 1]
    print("Raw LR AUC:", roc_auc_score(y_test_clean, y_prob_lr))
    print("Raw LR Log Loss:", log_loss(y_test_clean, y_prob_lr))
    print("Raw LR F1 Score:", f1_score(y_test_clean, y_pred_lr))
    
    # XGBoost with NaNs intact
    xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    print("Raw XGBoost AUC (all 235 rows):", roc_auc_score(y_test, y_prob_xgb))
    print("Raw XGBoost Log Loss:", log_loss(y_test, y_prob_xgb))
    print("Raw XGBoost F1 Score:", f1_score(y_test, y_pred_xgb))

if __name__ == "__main__":
    test_raw()
