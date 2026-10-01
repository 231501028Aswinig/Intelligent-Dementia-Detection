import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Style configuration for clean, publication-ready figures
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.autolayout'] = True

def generate_eda_plots(df, output_dir="results/visualizations"):
    """
    Generates all 5 required EDA visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []
    
    # 1. Age Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Age'], kde=True, bins=20, color='#2b5c8f', edgecolor='black', alpha=0.7)
    plt.title("Age Distribution of Study Cohort", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Age (Years)", fontsize=12)
    plt.ylabel("Patient Count", fontsize=12)
    path1 = os.path.join(output_dir, "eda_1_age_distribution.png")
    plt.savefig(path1, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(path1)
    
    # 2. Dementia Target Class Distribution
    plt.figure(figsize=(7, 5))
    target_series = df['CDR_binary'].map({0: 'Nondemented (0)', 1: 'Demented (1)'})
    counts = target_series.value_counts()
    colors = ['#2ca02c', '#d62728']
    bars = plt.bar(counts.index, counts.values, color=colors, edgecolor='black', width=0.5, alpha=0.85)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval} ({yval/len(df)*100:.1f}%)", ha='center', va='bottom', fontweight='bold')
    plt.title("Dementia Class Distribution (CDR Binary)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Clinical Diagnostic Class", fontsize=12)
    plt.ylabel("Number of Subjects", fontsize=12)
    plt.ylim(0, max(counts.values) * 1.15)
    path2 = os.path.join(output_dir, "eda_2_target_class_distribution.png")
    plt.savefig(path2, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(path2)
    
    # 3. MMSE Boxplot by Dementia Status
    plt.figure(figsize=(8, 5))
    plot_df = df.copy()
    plot_df['Status'] = plot_df['CDR_binary'].map({0: 'Nondemented', 1: 'Demented'})
    sns.boxplot(x='Status', y='MMSE', data=plot_df, hue='Status', palette={'Nondemented': '#6baed6', 'Demented': '#fc9272'}, width=0.4, boxprops=dict(alpha=0.9), legend=False)
    sns.stripplot(x='Status', y='MMSE', data=plot_df, color='black', alpha=0.3, jitter=0.2, size=5)
    plt.title("Mini-Mental State Examination (MMSE) by Dementia Status", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Clinical Status", fontsize=12)
    plt.ylabel("MMSE Score (0-30)", fontsize=12)
    path3 = os.path.join(output_dir, "eda_3_mmse_by_dementia_status.png")
    plt.savefig(path3, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(path3)
    
    # 4. Age vs MMSE Scatter Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x='Age', y='MMSE', hue='Status', style='Status', data=plot_df,
        palette={'Nondemented': '#1f77b4', 'Demented': '#d62728'},
        s=70, alpha=0.8
    )
    plt.title("Age vs. MMSE Score by Cognitive Diagnosis", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Age (Years)", fontsize=12)
    plt.ylabel("MMSE Score", fontsize=12)
    plt.axhline(24, color='gray', linestyle='--', alpha=0.6, label='MMSE Clinical Impairment Threshold (24)')
    plt.legend(title='Cognitive Group', loc='lower left')
    path4 = os.path.join(output_dir, "eda_4_age_vs_mmse_scatter.png")
    plt.savefig(path4, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(path4)
    
    # 5. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
        vmin=-1, vmax=1, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}
    )
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold', pad=12)
    path5 = os.path.join(output_dir, "eda_5_correlation_heatmap.png")
    plt.savefig(path5, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(path5)
    
    return generated_files

def generate_optimization_plots(results_df, optuna_res=None, pso_res=None, output_dir="results/visualizations"):
    """
    Generates optimization comparison and convergence plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []
    
    models = results_df['Model'].tolist()
    accuracies = results_df['Accuracy'].tolist()
    f1_scores = results_df['F1'].tolist()
    roc_aucs = results_df['ROC-AUC'].tolist()
    
    # 6. Baseline vs Optimized Accuracy Comparison
    plt.figure(figsize=(9, 5))
    palette = ['#7f7f7f'] + ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd'][:len(models)-1]
    bars = plt.bar(models, [acc * 100 for acc in accuracies], color=palette, edgecolor='black', width=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
    plt.title("Model Test Accuracy Comparison", fontsize=14, fontweight='bold', pad=12)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.ylim(0, 110)
    plt.xticks(rotation=15, ha='right')
    path6 = os.path.join(output_dir, "opt_6_accuracy_comparison.png")
    plt.savefig(path6, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(path6)
    
    # 7. Baseline vs Optimized F1-score Comparison
    plt.figure(figsize=(9, 5))
    bars = plt.bar(models, f1_scores, color=palette, edgecolor='black', width=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')
    plt.title("Model Test F1-Score Comparison", fontsize=14, fontweight='bold', pad=12)
    plt.ylabel("Test F1-Score", fontsize=12)
    plt.ylim(0, 1.15)
    plt.xticks(rotation=15, ha='right')
    path7 = os.path.join(output_dir, "opt_7_f1_comparison.png")
    plt.savefig(path7, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(path7)
    
    # 8. Baseline vs Optimized ROC-AUC Comparison
    plt.figure(figsize=(9, 5))
    bars = plt.bar(models, roc_aucs, color=palette, edgecolor='black', width=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')
    plt.title("Model Test ROC-AUC Comparison", fontsize=14, fontweight='bold', pad=12)
    plt.ylabel("Test ROC-AUC Score", fontsize=12)
    plt.ylim(0, 1.15)
    plt.xticks(rotation=15, ha='right')
    path8 = os.path.join(output_dir, "opt_8_roc_auc_comparison.png")
    plt.savefig(path8, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(path8)
    
    # 9. Comparison of All Optimization Methods Across Key Metrics
    plt.figure(figsize=(12, 6))
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'ROC-AUC']
    plot_data = []
    for _, row in results_df.iterrows():
        for m in metrics_to_plot:
            plot_data.append({'Model': row['Model'], 'Metric': m, 'Score': row[m]})
    m_df = pd.DataFrame(plot_data)
    
    sns.barplot(x='Metric', y='Score', hue='Model', data=m_df, palette='tab10')
    plt.title("Multi-Metric Performance Across All Optimization Techniques", fontsize=14, fontweight='bold', pad=12)
    plt.ylabel("Score (0 - 1.0)", fontsize=12)
    plt.ylim(0, 1.15)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    path9 = os.path.join(output_dir, "opt_9_all_methods_comparison.png")
    plt.savefig(path9, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(path9)
    
    # 10. Optuna Optimization History Plot
    if optuna_res and 'study' in optuna_res:
        study = optuna_res['study']
        trials = [t for t in study.trials if t.value is not None]
        trial_numbers = [t.number for t in trials]
        trial_values = [t.value for t in trials]
        best_values = [max(trial_values[:i+1]) for i in range(len(trial_values))]
        
        plt.figure(figsize=(8, 5))
        plt.scatter(trial_numbers, trial_values, color='#1f77b4', alpha=0.5, label='Trial CV Accuracy')
        plt.plot(trial_numbers, best_values, color='#d62728', linewidth=2.5, label='Best Objective Value')
        plt.title("Optuna Bayesian Optimization Convergence History", fontsize=14, fontweight='bold', pad=12)
        plt.xlabel("Trial Number", fontsize=12)
        plt.ylabel("5-Fold CV Accuracy", fontsize=12)
        plt.legend(loc='lower right')
        path10 = os.path.join(output_dir, "opt_10_optuna_optimization_history.png")
        plt.savefig(path10, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(path10)
        
        # 11. Optuna Hyperparameter Importance Plot
        try:
            import optuna.importance
            importances = optuna.importance.get_param_importances(study)
            if importances:
                plt.figure(figsize=(9, 5))
                p_names = list(importances.keys())
                p_vals = list(importances.values())
                # Sort ascending for horizontal bar chart
                p_names.reverse()
                p_vals.reverse()
                plt.barh(p_names, p_vals, color='#2ca02c', edgecolor='black', alpha=0.85)
                plt.title("Optuna Hyperparameter Relative Importance", fontsize=14, fontweight='bold', pad=12)
                plt.xlabel("Relative Importance Score", fontsize=12)
                path11 = os.path.join(output_dir, "opt_11_optuna_param_importances.png")
                plt.savefig(path11, dpi=300, bbox_inches='tight')
                plt.close()
                generated_files.append(path11)
        except Exception as e:
            print(f"Warning: Could not compute optuna importances: {e}")
            
    return generated_files

def generate_evaluation_plots(eval_results_list, X_train, y_train, X_test, y_test, feature_cols, output_dir="results/visualizations"):
    """
    Generates confusion matrices, ROC curves, PR curves, feature importance, and SHAP summary.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []
    
    # 12. Confusion Matrices for Models
    num_models = len(eval_results_list)
    cols = min(3, num_models)
    rows = (num_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), squeeze=False)
    for idx, eval_res in enumerate(eval_results_list):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        cm = eval_res['Confusion_Matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['Nondemented', 'Demented'], yticklabels=['Nondemented', 'Demented'],
                    annot_kws={"size": 14, "fontweight": "bold"})
        ax.set_title(f"{eval_res['Model']}\nAcc: {eval_res['Accuracy']*100:.1f}%, F1: {eval_res['F1']:.3f}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        
    # Hide unused axes
    for idx in range(num_models, rows * cols):
        r, c = divmod(idx, cols)
        fig.delaxes(axes[r][c])
        
    plt.tight_layout()
    path12 = os.path.join(output_dir, "eval_12_confusion_matrices.png")
    plt.savefig(path12, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(path12)
    
    # 13. ROC-AUC Curves for All Models on the Same Graph
    plt.figure(figsize=(8, 6))
    colors = ['#7f7f7f', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
    for idx, eval_res in enumerate(eval_results_list):
        if eval_res['y_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, eval_res['y_proba'])
            auc_val = eval_res['ROC-AUC']
            plt.plot(fpr, tpr, label=f"{eval_res['Model']} (AUC = {auc_val:.4f})",
                     color=colors[idx % len(colors)], linewidth=2.2)
                     
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Chance (AUC = 0.5000)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    plt.title("ROC-AUC Curves: Baseline vs. Optimized LightGBM Models", fontsize=14, fontweight='bold', pad=12)
    plt.legend(loc="lower right", frameon=True)
    path13 = os.path.join(output_dir, "eval_13_roc_curves_comparison.png")
    plt.savefig(path13, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(path13)
    
    # 14. Precision-Recall Curves
    plt.figure(figsize=(8, 6))
    for idx, eval_res in enumerate(eval_results_list):
        if eval_res['y_proba'] is not None:
            prec, rec, _ = precision_recall_curve(y_test, eval_res['y_proba'])
            pr_auc_val = eval_res['PR-AUC']
            plt.plot(rec, prec, label=f"{eval_res['Model']} (PR-AUC = {pr_auc_val:.4f})",
                     color=colors[idx % len(colors)], linewidth=2.2)
                     
    no_skill = (y_test == 1).sum() / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], 'k--', alpha=0.6, label=f'Baseline Chance ({no_skill:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (Sensitivity)", fontsize=12)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=12)
    plt.title("Precision-Recall Curves for All Models", fontsize=14, fontweight='bold', pad=12)
    plt.legend(loc="lower left", frameon=True)
    path14 = os.path.join(output_dir, "eval_14_precision_recall_curves.png")
    plt.savefig(path14, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append(path14)
    
    # 15. Feature Importance of Best Optimized Model
    # Select best model by F1 or ROC-AUC
    best_eval = max(eval_results_list, key=lambda x: (x['F1'], x['Accuracy'], x['ROC-AUC']))
    best_model = best_eval.get('model_obj', None)
    
    if best_model is not None and hasattr(best_model, "feature_importances_"):
        plt.figure(figsize=(9, 6))
        importances = best_model.feature_importances_
        indices = np.argsort(importances)
        plt.barh(range(len(indices)), importances[indices], color='#1f77b4', edgecolor='black', alpha=0.85)
        plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
        plt.xlabel("Feature Importance (Split / Frequency Count)", fontsize=12)
        plt.title(f"Feature Importance ({best_eval['Model']})", fontsize=14, fontweight='bold', pad=12)
        path15 = os.path.join(output_dir, "eval_15_feature_importance.png")
        plt.savefig(path15, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(path15)
        
    # 16. SHAP Summary Plot
    if best_model is not None:
        try:
            import shap
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test)
            
            # For binary classification, shap_values can be a list [vals_class0, vals_class1] or array
            if isinstance(shap_values, list):
                shap_to_plot = shap_values[1]
            else:
                shap_to_plot = shap_values
                
            plt.figure(figsize=(9, 6))
            shap.summary_plot(shap_to_plot, X_test, feature_names=feature_cols, show=False)
            plt.title(f"SHAP Summary Plot - Feature Impacts ({best_eval['Model']})", fontsize=13, fontweight='bold', pad=12)
            path16 = os.path.join(output_dir, "eval_16_shap_summary.png")
            plt.savefig(path16, dpi=300, bbox_inches='tight')
            plt.close()
            generated_files.append(path16)
        except Exception as e:
            print(f"Warning: SHAP summary plot generation encountered: {e}")
            
    return generated_files
