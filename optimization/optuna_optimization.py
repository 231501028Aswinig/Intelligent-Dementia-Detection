import time
import optuna
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

# Suppress verbose optuna logs during study
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_optuna_search(X_train, y_train, scale_pos_weight=1.0, cv_splits=5, n_trials=80, random_state=42):
    """
    Executes Optuna Bayesian optimization (TPE) for LightGBM
    using 5-Fold Stratified Cross-Validation on the training set only.
    """
    print("=" * 80)
    print(f"RUNNING OPTUNA (BAYESIAN TPE) OPTIMIZATION (LightGBM - {n_trials} trials)")
    print("=" * 80)
    
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.25, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 35),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 2.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 2.0, log=True),
            'scale_pos_weight': scale_pos_weight,
            'random_state': random_state,
            'n_jobs': 1,
            'verbose': -1
        }
        
        cv_scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            clf = LGBMClassifier(**params)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_val)
            cv_scores.append(accuracy_score(y_val, preds))
            
        return float(np.mean(cv_scores))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    
    start_time = time.time()
    study.optimize(objective, n_trials=n_trials)
    elapsed_time = time.time() - start_time
    
    best_params = study.best_params
    best_cv_score = study.best_value
    
    print(f"Optuna optimization completed in {elapsed_time:.2f}s")
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
        'method': 'Optuna',
        'best_model': best_model,
        'best_params': best_params,
        'best_cv_score': best_cv_score,
        'elapsed_time': elapsed_time,
        'study': study
    }
