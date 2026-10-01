"""
OASIS Dementia Baseline Model Training & Evaluation Script
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, log_loss, f1_score
import xgboost as xgb

def run_baseline_models():
    cleaned_path = os.path.join("data", "cleaned_oasis.csv")
    df = pd.read_csv(cleaned_path)
    
    # Drop non-feature ID column and target columns (CDR, CDR_binary)
    drop_cols = ['ID', 'CDR', 'CDR_binary']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]
    y = df['CDR_binary']
    
    # 80/20 Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    }
    
    results = []
    confusion_matrices = {}
    roc_data = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        results.append({
            'Model': name,
            'AUC-ROC': round(auc, 4),
            'Accuracy': round(acc, 4),
            'Loss': round(loss, 4),
            'F1 Score': round(f1, 4),
            'Sensitivity': round(sensitivity, 4),
            'Specificity': round(specificity, 4)
        })
        
        confusion_matrices[name] = cm
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, auc)
        
    results_df = pd.DataFrame(results)
    return results_df, confusion_matrices, roc_data, feature_cols

if __name__ == "__main__":
    df_results, cm_dict, roc_dict, features = run_baseline_models()
    print("Baseline Model Comparison Results:")
    print(df_results.to_string(index=False))
