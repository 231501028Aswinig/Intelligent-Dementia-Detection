import time
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from lightgbm import LGBMClassifier

def run_grid_search(X_train, y_train, scale_pos_weight=1.0, cv_splits=5, random_state=42):
    """
    Executes GridSearchCV hyperparameter optimization for LightGBM
    using 5-Fold Stratified Cross-Validation on the training set only.
    """
    print("=" * 80)
    print("RUNNING GRIDSEARCHCV OPTIMIZATION (LightGBM)")
    print("=" * 80)
    
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.03, 0.08],
        'num_leaves': [15, 31],
        'max_depth': [3, 6],
        'min_child_samples': [10, 20],
        'reg_alpha': [0.0, 0.5]
    }
    
    base_model = LGBMClassifier(
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        n_jobs=1,
        verbose=-1
    )
    
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=skf,
        n_jobs=1,
        verbose=0
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    
    print(f"GridSearchCV completed in {elapsed_time:.2f}s")
    print(f"Best 5-Fold CV Accuracy: {best_cv_score * 100:.2f}%")
    print("Best Hyperparameters:")
    for k, v in best_params.items():
        print(f"  - {k}: {v}")
        
    # Instantiate best model trained on full training set
    best_model = LGBMClassifier(
        **best_params,
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        n_jobs=1,
        verbose=-1
    )
    best_model.fit(X_train, y_train)
    
    return {
        'method': 'GridSearchCV',
        'best_model': best_model,
        'best_params': best_params,
        'best_cv_score': best_cv_score,
        'elapsed_time': elapsed_time,
        'search_object': grid_search
    }
