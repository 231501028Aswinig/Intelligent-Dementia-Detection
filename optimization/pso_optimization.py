import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

def map_vector_to_params(x, scale_pos_weight=1.0, random_state=42):
    """
    Maps continuous particle position vector (dim=9) to LightGBM hyperparameter dictionary.
    """
    params = {
        'n_estimators': int(np.clip(round(x[0]), 50, 300)),
        'learning_rate': float(np.clip(x[1], 0.01, 0.25)),
        'num_leaves': int(np.clip(round(x[2]), 15, 127)),
        'max_depth': int(np.clip(round(x[3]), 3, 9)),
        'min_child_samples': int(np.clip(round(x[4]), 5, 35)),
        'subsample': float(np.clip(x[5], 0.6, 1.0)),
        'colsample_bytree': float(np.clip(x[6], 0.6, 1.0)),
        'reg_alpha': float(np.clip(x[7], 0.0, 2.0)),
        'reg_lambda': float(np.clip(x[8], 0.0, 2.0)),
        'scale_pos_weight': scale_pos_weight,
        'random_state': random_state,
        'n_jobs': 1,
        'verbose': -1
    }
    return params

def run_pso_search(X_train, y_train, scale_pos_weight=1.0, cv_splits=5, n_particles=20, iters=20, random_state=42):
    """
    Executes Particle Swarm Optimization (PSO) for LightGBM
    using 5-Fold Stratified Cross-Validation on the training set only.
    """
    print("=" * 80)
    print(f"RUNNING PARTICLE SWARM OPTIMIZATION (PSO) (LightGBM - {n_particles} particles, {iters} iterations)")
    print("=" * 80)
    
    np.random.seed(random_state)
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    # Define hyperparameter bounds (lower and upper)
    # [n_est, lr, leaves, depth, min_child, subsample, colsample, reg_alpha, reg_lambda]
    lb = np.array([50, 0.01, 15, 3, 5, 0.6, 0.6, 0.0, 0.0])
    ub = np.array([300, 0.25, 127, 9, 35, 1.0, 1.0, 2.0, 2.0])
    dim = len(lb)
    
    # Memoization cache to avoid redundant model evaluations for identical discrete configs
    eval_cache = {}
    
    def evaluate_cv(param_vec):
        params = map_vector_to_params(param_vec, scale_pos_weight=scale_pos_weight, random_state=random_state)
        cache_key = tuple(sorted((k, v) for k, v in params.items() if k not in ['verbose']))
        if cache_key in eval_cache:
            return eval_cache[cache_key]
            
        cv_scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            clf = LGBMClassifier(**params)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_val)
            cv_scores.append(accuracy_score(y_val, preds))
            
        mean_acc = float(np.mean(cv_scores))
        eval_cache[cache_key] = mean_acc
        return mean_acc

    start_time = time.time()
    
    # Native Vectorized PSO implementation with cognitive (c1) & social (c2) acceleration and inertia weight (w)
    w = 0.729    # Inertia weight
    c1 = 1.49445 # Cognitive weight
    c2 = 1.49445 # Social weight
    
    # Initialize particle positions and velocities
    X = np.random.uniform(lb, ub, size=(n_particles, dim))
    V = np.random.uniform(-abs(ub - lb) * 0.1, abs(ub - lb) * 0.1, size=(n_particles, dim))
    
    pbest_X = X.copy()
    pbest_acc = np.zeros(n_particles)
    
    for i in range(n_particles):
        pbest_acc[i] = evaluate_cv(X[i])
        
    gbest_idx = np.argmax(pbest_acc)
    gbest_X = pbest_X[gbest_idx].copy()
    gbest_acc = pbest_acc[gbest_idx]
    
    pso_history = [gbest_acc]
    
    for iteration in range(1, iters + 1):
        r1 = np.random.rand(n_particles, dim)
        r2 = np.random.rand(n_particles, dim)
        
        # Velocity update
        V = w * V + c1 * r1 * (pbest_X - X) + c2 * r2 * (gbest_X - X)
        # Position update
        X = X + V
        
        # Boundary handling
        X = np.clip(X, lb, ub)
        
        # Evaluate fitness
        for i in range(n_particles):
            acc = evaluate_cv(X[i])
            if acc > pbest_acc[i]:
                pbest_acc[i] = acc
                pbest_X[i] = X[i].copy()
                if acc > gbest_acc:
                    gbest_acc = acc
                    gbest_X = X[i].copy()
                    
        pso_history.append(gbest_acc)
        
    elapsed_time = time.time() - start_time
    
    best_params = map_vector_to_params(gbest_X, scale_pos_weight=scale_pos_weight, random_state=random_state)
    # Remove non-tunable internal keys from best_params dict for display
    clean_params = {k: v for k, v in best_params.items() if k not in ['scale_pos_weight', 'random_state', 'verbose', 'subsample_freq']}
    
    print(f"PSO completed in {elapsed_time:.2f}s")
    print(f"Best 5-Fold CV Accuracy: {gbest_acc * 100:.2f}%")
    print("Best Hyperparameters:")
    for k, v in clean_params.items():
        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  - {k}: {val_str}")
        
    best_model = LGBMClassifier(
        **best_params
    )
    best_model.fit(X_train, y_train)
    
    return {
        'method': 'PSO',
        'best_model': best_model,
        'best_params': clean_params,
        'best_cv_score': gbest_acc,
        'elapsed_time': elapsed_time,
        'pso_history': pso_history
    }
