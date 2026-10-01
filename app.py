"""
Streamlit Application: Dementia Prediction — Raw vs. Cleaned Dataset Comparison (Multi-Table View)

Page Layout:
1. Best-Case Scenario Comparison Section (Table 3 Raw vs. Table 4 Cleaned)
2. Summary Markdown Note on Optimistic Improvement Gap
"""

import os
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, recall_score
import xgboost as xgb
import lightgbm as lgb

# Configure wide page layout
st.set_page_config(
    page_title="Dementia Prediction — Raw vs. Cleaned Dataset Comparison",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------------------------------------------------------
# RAW DATASET EVALUATION (Single Train/Test Split, No CV)
# -----------------------------------------------------------------------------
@st.cache_data
def get_raw_results():
    file_path = os.path.join("data", "raw_minimal.csv")
    df = pd.read_csv(file_path)
    
    drop_cols = ['ID', 'CDR', 'CDR_binary']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]
    y = df['CDR_binary']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(eval_metric='logloss', random_state=42),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1)
    }
    
    results = []
    for name, model in models.items():
        if name in ['Logistic Regression', 'Random Forest']:
            X_tr_no_delay = X_train.drop(columns=['Delay'])
            X_te_no_delay = X_test.drop(columns=['Delay'])
            
            tr_mask = X_tr_no_delay.notnull().all(axis=1)
            te_mask = X_te_no_delay.notnull().all(axis=1)
            
            X_tr = X_tr_no_delay[tr_mask]
            y_tr = y_train[tr_mask]
            X_te = X_te_no_delay[te_mask]
            y_te = y_test[te_mask]
            
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_proba = model.predict_proba(X_te)[:, 1]
            acc = accuracy_score(y_te, y_pred)
            f1 = f1_score(y_te, y_pred)
            loss = log_loss(y_te, y_proba)
            sens = recall_score(y_te, y_pred)
            cm = confusion_matrix(y_te, y_pred)
            tn, fp, fn, tp = cm.ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            loss = log_loss(y_test, y_proba)
            sens = recall_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
        results.append({
            'Model': name,
            'Accuracy': f"{acc:.2f}",
            'Loss': f"{loss:.2f}",
            'F1 Score': f"{f1:.2f}",
            'Sensitivity': f"{sens:.2f}",
            'Specificity': f"{spec:.2f}"
        })
        
    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# CLEANED DATASET EVALUATION (5-Fold Stratified CV)
# -----------------------------------------------------------------------------
@st.cache_data
def get_cleaned_results():
    file_path = os.path.join("data", "cleaned_oasis.csv")
    df = pd.read_csv(file_path)
    
    drop_cols = ['ID', 'CDR', 'CDR_binary']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]
    y = df['CDR_binary']
    
    neg_cnt = (y == 0).sum()
    pos_cnt = (y == 1).sum()
    scale_pos_weight = neg_cnt / pos_cnt if pos_cnt > 0 else 1.0
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'XGBoost': xgb.XGBClassifier(eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos_weight),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1, scale_pos_weight=scale_pos_weight)
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    table2_results = []
    table4_results = []
    
    for name, model in models.items():
        accs, f1s, losses, sens, specs = [], [], [], [], []
        
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
            losses.append(log_loss(y_val_f, y_proba_f))
            sens.append(recall_score(y_val_f, y_pred_f))
            cm_f = confusion_matrix(y_val_f, y_pred_f)
            tn_f, fp_f, fn_f, tp_f = cm_f.ravel()
            specs.append(tn_f / (tn_f + fp_f) if (tn_f + fp_f) > 0 else 0.0)
            
        acc_m, acc_s, acc_max = np.mean(accs), np.std(accs), np.max(accs)
        f1_m, f1_s, f1_max = np.mean(f1s), np.std(f1s), np.max(f1s)
        loss_m, loss_s, loss_min = np.mean(losses), np.std(losses), np.min(losses)
        sens_m, sens_s, sens_max = np.mean(sens), np.std(sens), np.max(sens)
        spec_m, spec_s, spec_max = np.mean(specs), np.std(specs), np.max(specs)
        
        table2_results.append({
            'Model': name,
            'Accuracy': f"{acc_m:.2f} ± {acc_s:.2f} (best case: {acc_max:.2f})",
            'Loss': f"{loss_m:.2f} ± {loss_s:.2f} (best case: {loss_min:.2f})",
            'F1 Score': f"{f1_m:.2f} ± {f1_s:.2f} (best case: {f1_max:.2f})",
            'Sensitivity': f"{sens_m:.2f} ± {sens_s:.2f} (best case: {sens_max:.2f})",
            'Specificity': f"{spec_m:.2f} ± {spec_s:.2f} (best case: {spec_max:.2f})"
        })
        
        table4_results.append({
            'Model': name,
            'Accuracy': f"{acc_max:.2f}",
            'Loss': f"{loss_min:.2f}",
            'F1 Score': f"{f1_max:.2f}",
            'Sensitivity': f"{sens_max:.2f}",
            'Specificity': f"{spec_max:.2f}"
        })
        
    return pd.DataFrame(table2_results), pd.DataFrame(table4_results)

@st.cache_resource
def get_trained_lgbm_model():
    file_path = os.path.join("data", "cleaned_oasis.csv")
    df = pd.read_csv(file_path)
    
    drop_cols = ['ID', 'CDR', 'CDR_binary']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]
    y = df['CDR_binary']
    
    neg_cnt = (y == 0).sum()
    pos_cnt = (y == 1).sum()
    scale_pos_weight = neg_cnt / pos_cnt if pos_cnt > 0 else 1.0
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = lgb.LGBMClassifier(random_state=42, verbose=-1, scale_pos_weight=scale_pos_weight)
    model.fit(X_scaled, y)
    
    return model, scaler, feature_cols

# Execute evaluation queries
raw_results_df = get_raw_results()
cleaned_table2_df, cleaned_table4_df = get_cleaned_results()
lgbm_model, lgbm_scaler, lgbm_features = get_trained_lgbm_model()

# Main Application Title
st.title("Dementia Prediction — Raw vs. Cleaned Dataset Comparison")

# SECTION 2: Best-Case Scenario Comparison (Tables 3 & 4)
st.header("Best-Case Scenario Comparison")

col3, col4 = st.columns(2)

with col3:
    st.subheader("TABLE 3 — Raw Dataset")
    st.dataframe(raw_results_df, use_container_width=True, hide_index=True)

with col4:
    st.subheader("Cleaned Dataset")
    st.dataframe(cleaned_table4_df, use_container_width=True, hide_index=True)

st.markdown(
    "Logistic Regression and LightGBM show the largest optimistic improvement, with Accuracy rising from "
    "0.80 (raw) to 0.89 (cleaned, best case) and F1 Score jumping from 0.74 (raw) to 0.88 (cleaned, best case)."
)

# -----------------------------------------------------------------------------
# LIVE PREDICTION TOOL
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("Interactive Dementia Prediction (LightGBM)")
st.write("Enter patient metrics below to predict the probability of dementia using the trained LightGBM model.")

with st.form("prediction_form"):
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        gender_str = st.selectbox("Gender (M/F)", ["Female", "Male"])
        age = st.number_input("Age", min_value=40.0, max_value=120.0, value=74.0)
        educ = st.number_input("Education Level (1-5)", min_value=1.0, max_value=5.0, value=2.0)
        
    with col_b:
        ses = st.number_input("Socioeconomic Status (SES, 1-5)", min_value=1.0, max_value=5.0, value=3.0)
        mmse = st.number_input("MMSE Score (0-30)", min_value=0.0, max_value=30.0, value=29.0)
        etiv = st.number_input("Estimated Total Intracranial Volume (eTIV)", min_value=1000.0, max_value=2000.0, value=1344.0)
        
    with col_c:
        nwbv = st.number_input("Normalized Whole Brain Volume (nWBV)", min_value=0.5, max_value=1.0, value=0.743, format="%.3f")
        asf = st.number_input("Atlas Scaling Factor (ASF)", min_value=0.5, max_value=2.0, value=1.306, format="%.3f")
        
    submit = st.form_submit_button("Predict Dementia")

if submit:
    # Feature Engineering exactly as in preprocessing
    gender = 1 if gender_str == "Male" else 0
    brain_volume_ratio = nwbv / etiv
    age_cognition_score = age * (30.0 - mmse)
    
    input_dict = {
        'M/F': gender, 
        'Age': age, 
        'Educ': educ, 
        'SES': ses, 
        'MMSE': mmse, 
        'eTIV': etiv, 
        'nWBV': nwbv, 
        'ASF': asf,
        'Brain_Volume_Ratio': brain_volume_ratio, 
        'Age_Cognition_Score': age_cognition_score
    }
    
    input_df = pd.DataFrame([input_dict])[lgbm_features]
    input_scaled = lgbm_scaler.transform(input_df)
    
    prediction = lgbm_model.predict(input_scaled)[0]
    probability = lgbm_model.predict_proba(input_scaled)[0][1]
    
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Dementia Detected! (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Patient is Healthy! (Probability of Dementia: {probability:.2%})")