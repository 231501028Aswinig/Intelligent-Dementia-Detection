import os
import time
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

from .data_loader import load_and_preprocess_data, get_stratified_split
from .grid_search import run_grid_search
from .randomized_search import run_randomized_search
from .optuna_optimization import run_optuna_search
from .pso_optimization import run_pso_search
from .evaluation import evaluate_model_performance
from .visualizations import generate_eda_plots, generate_optimization_plots, generate_evaluation_plots

def execute_full_optimization_pipeline(raw_path="dementia_dataset_2.csv",
                                      cleaned_path="data/cleaned_dataset_2.csv",
                                      output_dir="results/visualizations",
                                      results_csv="optimization_results_without_smote.csv"):
    """
    Orchestrates the full hyperparameter optimization, model comparison,
    comprehensive evaluation, and visualization pipeline without SMOTE or data leakage.
    """
    print("=" * 90)
    print("DEMENTIA PREDICTION: MODEL OPTIMIZATION & COMPREHENSIVE EVALUATION PIPELINE")
    print("=" * 90)
    print("Strict Protocol Active:")
    print("  - NO SMOTE / Synthetic Oversampling")
    print("  - Stratified 80:20 Train/Test Split (Locked Test Set)")
    print("  - 5-Fold Stratified Cross-Validation strictly on Training Data")
    print("  - Single locked test-set evaluation after optimization")
    print("=" * 90)

    # 1. Load data and run existing preprocessing & feature engineering
    df, feature_cols, target_col = load_and_preprocess_data(raw_path, cleaned_path)
    print(f"Cleaned dataset loaded: {df.shape[0]} samples, {len(feature_cols)} features.")
    print(f"Features: {feature_cols}")
    
    # 2. Generate EDA Visualizations
    print("\n--- STEP 1: Generating EDA Visualizations ---")
    eda_files = generate_eda_plots(df, output_dir=output_dir)
    for f in eda_files:
        print(f"  [Saved] {f}")
        
    # 3. Stratified 80:20 Train/Test Split & RobustScaler fit strictly on Train
    print("\n--- STEP 2: Creating Stratified 80:20 Train-Test Split ---")
    split_data = get_stratified_split(df, feature_cols, target_col=target_col, test_size=0.20, random_state=42)
    
    X_train = split_data['X_train_scaled']
    X_test = split_data['X_test_scaled']
    y_train = split_data['y_train']
    y_test = split_data['y_test']
    scale_pos_weight = split_data['scale_pos_weight']
    
    print(f"Training set: {len(X_train)} samples ({y_train.sum()} positive, {(y_train == 0).sum()} negative)")
    print(f"Locked Test set: {len(X_test)} samples ({y_test.sum()} positive, {(y_test == 0).sum()} negative)")
    print(f"Class imbalance weighting: scale_pos_weight = {scale_pos_weight:.2f}")
    
    # 4. Train Baseline LightGBM
    print("\n--- STEP 3: Training Baseline LightGBM (Default Settings) ---")
    baseline_lgbm = LGBMClassifier(
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        n_jobs=1,
        verbose=-1
    )
    baseline_lgbm.fit(X_train, y_train)
    
    # Compute 5-fold CV for baseline for fair CV comparison
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    baseline_cv_scores = cross_val_score(baseline_lgbm, X_train, y_train, cv=skf, scoring='accuracy')
    baseline_cv_mean = float(np.mean(baseline_cv_scores))
    print(f"Baseline LightGBM 5-Fold CV Accuracy: {baseline_cv_mean * 100:.2f}%")

    # 5. Hyperparameter Optimization on Training Data Only
    print("\n--- STEP 4: Running 4 Optimization Techniques (5-Fold CV on Training Set) ---")
    
    # Method 1: GridSearchCV
    grid_res = run_grid_search(X_train, y_train, scale_pos_weight=scale_pos_weight, cv_splits=5, random_state=42)
    
    # Method 2: RandomizedSearchCV
    rand_res = run_randomized_search(X_train, y_train, scale_pos_weight=scale_pos_weight, cv_splits=5, n_iter=60, random_state=42)
    
    # Method 3: Optuna (Bayesian TPE)
    optuna_res = run_optuna_search(X_train, y_train, scale_pos_weight=scale_pos_weight, cv_splits=5, n_trials=50, random_state=42)
    
    # Method 4: Particle Swarm Optimization (PSO)
    pso_res = run_pso_search(X_train, y_train, scale_pos_weight=scale_pos_weight, cv_splits=5, n_particles=15, iters=15, random_state=42)

    # 6. Comprehensive Evaluation on Locked Test Set
    print("\n--- STEP 5: Final Evaluation on Locked Test Set (Evaluated Once) ---")
    models_to_eval = [
        ('Existing Baseline LightGBM', baseline_lgbm, baseline_cv_mean, 'Baseline'),
        ('GridSearch LightGBM', grid_res['best_model'], grid_res['best_cv_score'], 'GridSearchCV'),
        ('RandomizedSearch LightGBM', rand_res['best_model'], rand_res['best_cv_score'], 'RandomizedSearchCV'),
        ('Optuna LightGBM', optuna_res['best_model'], optuna_res['best_cv_score'], 'Optuna (TPE)'),
        ('PSO LightGBM', pso_res['best_model'], pso_res['best_cv_score'], 'PSO')
    ]
    
    eval_results_list = []
    summary_rows = []
    
    for name, model_obj, cv_score, method_label in models_to_eval:
        res = evaluate_model_performance(model_obj, X_test, y_test, model_name=name)
        res['model_obj'] = model_obj
        res['CV_Score'] = cv_score
        res['Method'] = method_label
        eval_results_list.append(res)
        
        summary_rows.append({
            'Model': name,
            'CV_Accuracy': round(cv_score, 4),
            'Accuracy': round(res['Accuracy'], 4),
            'Precision': round(res['Precision'], 4),
            'Recall': round(res['Recall'], 4),
            'Specificity': round(res['Specificity'], 4),
            'F1': round(res['F1'], 4),
            'ROC-AUC': round(res['ROC-AUC'], 4),
            'PR-AUC': round(res['PR-AUC'], 4)
        })
        
    results_df = pd.DataFrame(summary_rows)
    
    # Print Comparison Table
    print("\n" + "=" * 90)
    print("FINAL COMPARISON TABLE (LOCKED TEST SET PERFORMANCE)")
    print("=" * 90)
    display_cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'ROC-AUC', 'PR-AUC']
    print(results_df[display_cols].to_string(index=False))
    print("=" * 90)

    # Save to CSV
    results_df.to_csv(results_csv, index=False)
    print(f"\n[Saved] Final results table to: {results_csv}")
    
    # Also save a copy inside results/
    os.makedirs("results", exist_ok=True)
    results_df.to_csv(os.path.join("results", results_csv), index=False)

    # 7. Generate Model Optimization & Final Evaluation Plots
    print("\n--- STEP 6: Generating Optimization and Final Evaluation Plots ---")
    opt_files = generate_optimization_plots(results_df, optuna_res=optuna_res, pso_res=pso_res, output_dir=output_dir)
    for f in opt_files:
        print(f"  [Saved] {f}")
        
    eval_files = generate_evaluation_plots(eval_results_list, X_train, y_train, X_test, y_test, feature_cols, output_dir=output_dir)
    for f in eval_files:
        print(f"  [Saved] {f}")

    all_generated_plots = eda_files + opt_files + eval_files
    
    print("\n" + "=" * 90)
    print(f"PIPELINE COMPLETE: {len(all_generated_plots)} visualizations generated.")
    print("=" * 90)
    
    return {
        'results_df': results_df,
        'eval_results_list': eval_results_list,
        'grid_res': grid_res,
        'rand_res': rand_res,
        'optuna_res': optuna_res,
        'pso_res': pso_res,
        'generated_plots': all_generated_plots
    }
