import time
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from scipy.stats import uniform, randint
from lightgbm import LGBMClassifier

def run_randomized_search(X_train, y_train, scale_pos_weight=1.0, cv_splits=5, n_iter=100, random_state=42):
    """
    Executes RandomizedSearchCV hyperparameter optimization for LightGBM
    using 5-Fold Stratified Cross-Validation on the training set only.
    """
    print("=" * 80)
    print(f"RUNNING RANDOMIZEDSEARCHCV OPTIMIZATION (LightGBM - {n_iter} iterations)")
    print("=" * 80)
    
    param_distributions = {
        'n_estimators': randint(50, 300),
        'learning_rate': uniform(0.01, 0.25),
        'num_leaves': randint(15, 127),
        'max_depth': [-1, 3, 4, 5, 6, 7, 8, 9],
        'min_child_samples': randint(5, 35),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'reg_alpha': uniform(0.0, 2.0),
        'reg_lambda': uniform(0.0, 2.0)
    }
    
    base_model = LGBMClassifier(
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        n_jobs=1,
        verbose=-1
    )
    
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring='accuracy',
        cv=skf,
        random_state=random_state,
        n_jobs=1,
        verbose=0
    )
    
    start_time = time.time()
    random_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    best_params = random_search.best_params_
    best_cv_score = random_search.best_score_
    
    print(f"RandomizedSearchCV completed in {elapsed_time:.2f}s")
    print(f"Best 5-Fold CV Accuracy: {best_cv_score * 100:.2f}%")
    print("Best Hyperparameters:")
    for k, v in best_params.items():
        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  - {k}: {val_str}")
        
    best_model = LGBMClassifier(
        **best_params,
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        n_jobs=1,
        verbose=-1
    )
    best_model.fit(X_train, y_train)
    
    return {
        'method': 'RandomizedSearchCV',
        'best_model': best_model,
        'best_params': best_params,
        'best_cv_score': best_cv_score,
        'elapsed_time': elapsed_time,
        'search_object': random_search
    }
