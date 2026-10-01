"""
Optimization and Comprehensive Evaluation Module for Dementia Prediction.
"""
from .data_loader import load_and_preprocess_data, get_stratified_split
from .grid_search import run_grid_search
from .randomized_search import run_randomized_search
from .optuna_optimization import run_optuna_search
from .pso_optimization import run_pso_search
from .evaluation import evaluate_model_performance, compute_all_metrics
from .visualizations import generate_eda_plots, generate_optimization_plots, generate_evaluation_plots
