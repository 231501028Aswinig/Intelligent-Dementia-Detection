"""
Root execution script for the Dementia Prediction Optimization & Evaluation Module.
"""
from optimization.pipeline import execute_full_optimization_pipeline

if __name__ == "__main__":
    results = execute_full_optimization_pipeline(
        raw_path="dementia_dataset_2.csv",
        cleaned_path="data/cleaned_dataset_2.csv",
        output_dir="results/visualizations",
        results_csv="optimization_results_without_smote.csv"
    )
