import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)

def compute_all_metrics(y_true, y_pred, y_proba=None):
    """
    Computes all standard clinical and ML evaluation metrics.
    
    Returns:
        dict with Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, PR-AUC, CM
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        tn = fp = fn = tp = 0
        spec = 0.0
        
    roc_auc = roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan
    pr_auc = average_precision_score(y_true, y_proba) if y_proba is not None else np.nan
    
    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'Specificity': spec,
        'F1': f1,
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc,
        'Confusion_Matrix': cm,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp
    }

def evaluate_model_performance(model, X_test, y_test, model_name="Model"):
    """
    Evaluates a trained model on the locked test set.
    """
    y_pred = model.predict(X_test)
    
    # Predict probabilities if supported
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = None
        
    metrics = compute_all_metrics(y_test, y_pred, y_proba)
    metrics['Model'] = model_name
    metrics['Classification_Report'] = classification_report(y_test, y_pred, zero_division=0)
    metrics['y_pred'] = y_pred
    metrics['y_proba'] = y_proba
    
    return metrics
