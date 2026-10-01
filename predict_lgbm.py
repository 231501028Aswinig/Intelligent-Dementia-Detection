import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler

def get_trained_lgbm_model(data_path="data/cleaned_oasis.csv"):
    # Load data
    df = pd.read_csv(data_path)
    
    drop_cols = ['ID', 'CDR', 'CDR_binary']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]
    y = df['CDR_binary']
    
    # Calculate scale_pos_weight
    neg_cnt = (y == 0).sum()
    pos_cnt = (y == 1).sum()
    scale_pos_weight = neg_cnt / pos_cnt if pos_cnt > 0 else 1.0
    
    # Fit scaler on full data
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model on full data
    model = lgb.LGBMClassifier(random_state=42, verbose=-1, scale_pos_weight=scale_pos_weight)
    model.fit(X_scaled, y)
    
    return model, scaler, feature_cols

def predict_dementia(input_data_dict, model, scaler, feature_cols):
    """
    input_data_dict: Dictionary containing the feature values.
    """
    # Create DataFrame from input
    input_df = pd.DataFrame([input_data_dict])[feature_cols]
    
    # Scale input
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    return prediction, probability

if __name__ == "__main__":
    print("Training LightGBM model on full dataset...")
    model, scaler, feature_cols = get_trained_lgbm_model()
    
    # Example 1: Healthy Patient (similar to OAS1_0001_MR1)
    sample_healthy = {
        'M/F': 0, 'Age': 74.0, 'Educ': 2.0, 'SES': 3.0, 
        'MMSE': 29.0, 'eTIV': 1344, 'nWBV': 0.743, 'ASF': 1.306,
        'Brain_Volume_Ratio': 0.000552827380952381, 'Age_Cognition_Score': 74.0
    }
    
    # Example 2: Dementia Patient (similar to OAS1_0003_MR1)
    sample_dementia = {
        'M/F': 0, 'Age': 73.0, 'Educ': 4.0, 'SES': 3.0, 
        'MMSE': 27.0, 'eTIV': 1454, 'nWBV': 0.708, 'ASF': 1.207,
        'Brain_Volume_Ratio': 0.0004869325997248968, 'Age_Cognition_Score': 219.0
    }
    
    # Predict Healthy
    pred1, prob1 = predict_dementia(sample_healthy, model, scaler, feature_cols)
    print("\n--- Prediction for Sample Healthy Input ---")
    print(f"Prediction: {'DEMENTIA (1)' if pred1 == 1 else 'HEALTHY (0)'}")
    print(f"Probability of Dementia: {prob1:.2%}")
    
    # Predict Dementia
    pred2, prob2 = predict_dementia(sample_dementia, model, scaler, feature_cols)
    print("\n--- Prediction for Sample Dementia Input ---")
    print(f"Prediction: {'DEMENTIA (1)' if pred2 == 1 else 'HEALTHY (0)'}")
    print(f"Probability of Dementia: {prob2:.2%}")


    my_patient_input = {
        'M/F': 0,                        # 0 for Female, 1 for Male
        'Age': 74.0, 
        'Educ': 2.0, 
        'SES': 3.0, 
        'MMSE': 29.0, 
        'eTIV': 1344, 
        'nWBV': 0.743, 
        'ASF': 1.306,
        'Brain_Volume_Ratio': 0.000552,  # Usually eTIV / (nWBV * 1000000) or similar ratio
        'Age_Cognition_Score': 74.0      # E.g. (100 - Age) * MMSE 
    }
    
    # Predict for Your Patient Input
    pred_mine, prob_mine = predict_dementia(my_patient_input, model, scaler, feature_cols)
    print("\n--- Prediction for MY Patient Input ---")
    print(f"Prediction: {'DEMENTIA (1)' if pred_mine == 1 else 'HEALTHY (0)'}")
    print(f"Probability of Dementia: {prob_mine:.2%}")
