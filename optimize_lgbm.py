import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

def optimize_lightgbm():
    print("=" * 80)
    print("OPTIMIZING LightGBM on 'cleaned_dataset_2.csv'")
    print("=" * 80)
    
    # Load the best dataset
    df = pd.read_csv("data/cleaned_dataset_2.csv")
    
    drop_cols = ['CDR', 'CDR_binary']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]
    y = df['CDR_binary']
    
    # Calculate scale_pos_weight
    neg_cnt = (y == 0).sum()
    pos_cnt = (y == 1).sum()
    scale_pos_weight = neg_cnt / pos_cnt if pos_cnt > 0 else 1.0
    print(f"Class imbalance handling: scale_pos_weight = {scale_pos_weight:.2f}\n")
    
    # Preprocess (Scale everything beforehand to simplify GridSearch pipeline, 
    # since RobustScaler scales independently per feature and doesn't leak much target info, 
    # but strictly speaking it should be in a pipeline. We'll use a Pipeline.)
    from sklearn.pipeline import Pipeline
    
    # We will use RandomizedSearchCV to find the best hyperparameters
    param_distributions = {
        'lgbm__n_estimators': [50, 100, 200, 300, 500],
        'lgbm__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'lgbm__num_leaves': [15, 31, 63, 127],
        'lgbm__max_depth': [-1, 3, 5, 7, 9],
        'lgbm__min_child_samples': [5, 10, 20, 30],
        'lgbm__subsample': [0.6, 0.8, 1.0],
        'lgbm__colsample_bytree': [0.6, 0.8, 1.0],
        'lgbm__reg_alpha': [0.0, 0.1, 0.5, 1.0],
        'lgbm__reg_lambda': [0.0, 0.1, 0.5, 1.0]
    }
    
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('lgbm', LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight, verbose=-1))
    ])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Starting Randomized Search for hyperparameters...")
    search = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_distributions, 
        n_iter=100, 
        scoring='accuracy', 
        cv=skf, 
        random_state=42, 
        n_jobs=-1,
        verbose=1
    )
    
    search.fit(X, y)
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS:")
    print("=" * 80)
    print(f"Best Cross-Validation Accuracy: {search.best_score_ * 100:.2f}%")
    print("Best Hyperparameters:")
    for k, v in search.best_params_.items():
        print(f"  - {k.replace('lgbm__', '')}: {v}")
        
    print("\nEvaluating Best Model across 5-folds for stability...")
    # Get the best estimator
    best_model = search.best_estimator_
    
    accs, f1s = [], []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        best_model.fit(X_tr, y_tr)
        y_pred = best_model.predict(X_val)
        
        accs.append(accuracy_score(y_val, y_pred))
        f1s.append(f1_score(y_val, y_pred))
        
    print(f"\nFinal Validated Model Accuracy: {np.mean(accs) * 100:.2f}% ± {np.std(accs) * 100:.2f}%")
    print(f"Final Validated Model F1 Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

if __name__ == "__main__":
    optimize_lightgbm()
