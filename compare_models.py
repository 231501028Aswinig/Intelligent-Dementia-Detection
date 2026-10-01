"""
STAGE 3: Comprehensive Model Evaluation Script (Raw vs. Cleaned Dataset)

Evaluates four machine learning models (Logistic Regression, Random Forest, XGBoost, LightGBM)
across two dataset variants:

1. RAW Minimal Baseline ('data/raw_minimal.csv'):
   - Single 80/20 train/test split, unscaled features, no class weighting, no CV.
   - Logistic Regression & Random Forest drop NaN rows (19 SES-NaN rows dropped).
   - XGBoost & LightGBM utilize NaNs natively (all 235 rows used).

2. CLEANED Refined Pipeline ('data/cleaned_oasis.csv'):
   - 5-Fold Stratified Cross-Validation (mean ± std reported across folds).
   - RobustScaler feature scaling (resilient to residual extreme values).
   - Class weighting (class_weight='balanced' for LR/RF, scale_pos_weight=1.35 for XGB/LGBM).
   - Single 80/20 held-out test set evaluation for confusion matrix plots.
"""

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

def evaluate_raw_dataset(file_path):
    df = pd.read_csv(file_path)
    
    drop_cols = ['ID', 'CDR', 'CDR_binary']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]
    y = df['CDR_binary']
    
    # Single 80/20 Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Models without scaling or class weighting
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(eval_metric='logloss', random_state=42),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1)
    }
    
    results = []
    confusion_matrices = {}
    rows_used_info = {}
    
    for name, model in models.items():
        if name in ['Logistic Regression', 'Random Forest']:
            # Drop 100% NaN 'Delay' column and drop NaN rows in SES
            X_train_no_delay = X_train.drop(columns=['Delay'])
            X_test_no_delay = X_test.drop(columns=['Delay'])
            
            train_mask = X_train_no_delay.notnull().all(axis=1)
            test_mask = X_test_no_delay.notnull().all(axis=1)
            
            X_tr = X_train_no_delay[train_mask]
            y_tr = y_train[train_mask]
            X_te = X_test_no_delay[test_mask]
            y_te = y_test[test_mask]
            
            dropped_cnt = (~train_mask).sum() + (~test_mask).sum()
            used_cnt = len(X_tr) + len(X_te)
            rows_used_info[name] = f"{used_cnt} rows (dropped {dropped_cnt} NaN rows)"
            
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_proba = model.predict_proba(X_te)[:, 1]
            
            acc = accuracy_score(y_te, y_pred)
            f1 = f1_score(y_te, y_pred)
            loss = log_loss(y_te, y_proba)
            sensitivity = recall_score(y_te, y_pred)
            
            cm = confusion_matrix(y_te, y_pred)
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            auc = roc_auc_score(y_te, y_proba)
        else:
            # XGBoost and LightGBM handle NaNs natively across all 235 rows
            rows_used_info[name] = f"{len(X_train)+len(X_test)} rows (used NaNs natively)"
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            loss = log_loss(y_test, y_proba)
            sensitivity = recall_score(y_test, y_pred)
            
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            auc = roc_auc_score(y_test, y_proba)
            
        results.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Loss': round(loss, 4),
            'F1 Score': round(f1, 4),
            'Sensitivity': round(sensitivity, 4),
            'Specificity': round(specificity, 4),
            'AUC-ROC': round(auc, 4),
            'Rows Used': rows_used_info[name]
        })
        
        confusion_matrices[name] = cm
        
    return pd.DataFrame(results), confusion_matrices, rows_used_info

def evaluate_cleaned_dataset(file_path):
    df = pd.read_csv(file_path)
    
    drop_cols = ['ID', 'CDR', 'CDR_binary']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]
    y = df['CDR_binary']
    
    # Calculate ratio for scale_pos_weight: neg_count / pos_count
    neg_cnt = (y == 0).sum()
    pos_cnt = (y == 1).sum()
    scale_pos_weight = neg_cnt / pos_cnt if pos_cnt > 0 else 1.0
    
    # Models with class weighting
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'XGBoost': xgb.XGBClassifier(eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos_weight),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1, scale_pos_weight=scale_pos_weight)
    }
    
    # 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    
    for name, model in models.items():
        accs, f1s, sens, specs, aucs, losses = [], [], [], [], [], []
        
        for train_idx, val_idx in skf.split(X, y):
            X_tr_f, X_val_f = X.iloc[train_idx], X.iloc[val_idx]
            y_tr_f, y_val_f = y.iloc[train_idx], y.iloc[val_idx]
            
            scaler = RobustScaler()
            X_tr_scaled = scaler.fit_transform(X_tr_f)
            X_val_scaled = scaler.transform(X_val_f)
            
            model.fit(X_tr_scaled, y_tr_f)
            y_pred_f = model.predict(X_val_scaled)
            y_proba_f = model.predict_proba(X_val_scaled)[:, 1]
            
            accs.append(accuracy_score(y_val_f, y_pred_f))
            f1s.append(f1_score(y_val_f, y_pred_f))
            sens.append(recall_score(y_val_f, y_pred_f))
            
            cm_f = confusion_matrix(y_val_f, y_pred_f)
            tn_f, fp_f, fn_f, tp_f = cm_f.ravel()
            specs.append(tn_f / (tn_f + fp_f) if (tn_f + fp_f) > 0 else 0.0)
            
            aucs.append(roc_auc_score(y_val_f, y_proba_f))
            losses.append(log_loss(y_val_f, y_proba_f))
            
        acc_mean, acc_std = np.mean(accs), np.std(accs)
        f1_mean, f1_std = np.mean(f1s), np.std(f1s)
        sens_mean, sens_std = np.mean(sens), np.std(sens)
        spec_mean, spec_std = np.mean(specs), np.std(specs)
        auc_mean, auc_std = np.mean(aucs), np.std(aucs)
        loss_mean, loss_std = np.mean(losses), np.std(losses)
        
        results.append({
            'Model': name,
            'Accuracy': round(acc_mean, 4),
            'Accuracy (std)': round(acc_std, 4),
            'Loss': round(loss_mean, 4),
            'Loss (std)': round(loss_std, 4),
            'F1 Score': round(f1_mean, 4),
            'F1 Score (std)': round(f1_std, 4),
            'Sensitivity': round(sens_mean, 4),
            'Sensitivity (std)': round(sens_std, 4),
            'Specificity': round(spec_mean, 4),
            'Specificity (std)': round(spec_std, 4),
            'AUC-ROC': round(auc_mean, 4),
            'AUC-ROC (std)': round(auc_std, 4)
        })
        
    # Held-out 80/20 test split for confusion matrix display
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    confusion_matrices = {}
    
    for name, model in models.items():
        scaler = RobustScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)
        
        model.fit(X_tr_s, y_train)
        y_pred = model.predict(X_te_s)
        confusion_matrices[name] = confusion_matrix(y_test, y_pred)
        
    return pd.DataFrame(results), confusion_matrices

def run_all_evaluations():
    raw_path = os.path.join("data", "raw_minimal.csv")
    cleaned_path = os.path.join("data", "cleaned_oasis.csv")
    
    df_raw_results, cm_raw, rows_info = evaluate_raw_dataset(raw_path)
    df_cleaned_results, cm_cleaned = evaluate_cleaned_dataset(cleaned_path)
    
    print("=" * 80)
    print("RAW DATASET MODEL RESULTS ('data/raw_minimal.csv' - Unscaled, No CV, No Imputation):")
    print("=" * 80)
    print(df_raw_results.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("CLEANED DATASET MODEL RESULTS ('data/cleaned_oasis.csv' - 5-Fold CV, RobustScaler, Class-Weighted):")
    print("=" * 80)
    print(df_cleaned_results.to_string(index=False))
    
    return {
        'raw': {'df': df_raw_results, 'cm': cm_raw, 'rows_info': rows_info},
        'cleaned': {'df': df_cleaned_results, 'cm': cm_cleaned}
    }

if __name__ == "__main__":
    run_all_evaluations()
