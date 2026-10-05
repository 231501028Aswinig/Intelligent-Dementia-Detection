# Intelligent Dementia Detection & Machine Learning Optimization

An end-to-end Machine Learning pipeline and research benchmark for early prediction and classification of Dementia using patient clinical, cognitive, and MRI volumetric data.

---

## 📌 Project Overview
This project presents an optimized machine learning pipeline designed to predict cognitive impairment and dementia status (Clinical Dementia Rating $\ge 0.5$). 

### Key Highlights:
- **High Performance**: Achieves **93.33% accuracy** and **0.93 F1-score** on unseen holdout validation data using an optimized Extra Trees + SMOTE pipeline.
- **Hyperparameter Optimization Suite**: Comprehensive evaluation comparing **Particle Swarm Optimization (PSO)**, **Optuna (Bayesian TPE)**, **GridSearchCV**, and **RandomizedSearchCV**.
- **Interpretability & Visualizations**: In-depth Exploratory Data Analysis (EDA), SHAP summary explainability, ROC curves, Precision-Recall curves, and Confusion Matrices.
- **Interactive Web Application**: Streamlit dashboard (`app.py`) for live prediction and risk stratification.

---

## 📊 Performance Summary

| Metric | Baseline (Cross-Sectional) | Longitudinal Baseline | Final Optimized Pipeline |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 82.13% | 83.91% | **93.33%** |
| **Macro F1-Score** | 0.7833 | 0.8148 | **0.9300** |
| **Sensitivity (Recall)**| 0.7700 | 0.7963 | **0.9300** |
| **Precision (Dementia)**| 0.8100 | 0.8300 | **0.9700** |

---

## 🛠️ Project Structure

```text
├── data/
│   └── cleaned_dataset_2.csv       # Preprocessed longitudinal dataset
├── optimization/                   # Optimization and evaluation modules
│   ├── data_loader.py              # Robust data loading & split
│   ├── evaluation.py               # Multi-metric evaluation framework
│   ├── grid_search.py              # Exhaustive Grid Search
│   ├── randomized_search.py        # Randomized Hyperparameter Search
│   ├── optuna_optimization.py      # Optuna Bayesian optimization
│   ├── pso_optimization.py         # Particle Swarm Optimization (PSO)
│   ├── pipeline.py                 # Pipeline runner
│   └── visualizations.py           # Chart generation utilities
├── outputs/                        # High-resolution output plots & figures
├── results/                        # Benchmark results & metrics CSVs
├── app.py                          # Streamlit interactive application
├── preprocessing.py                # Data cleaning & feature engineering
├── final_best_model.py             # Final validated Extra Trees pipeline
├── final_project_report.md         # Comprehensive project research report
├── Intelligent_Dementia_Detection_Report.docx  # Full project report document
└── README.md                       # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation
Ensure you have Python 3.10+ installed.

```bash
# Clone the repository
git clone https://github.com/231501028Aswinig/Intelligent-Dementia-Detection.git
cd Intelligent-Dementia-Detection

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install numpy pandas scikit-learn lightgbm xgboost optuna pyswarm matplotlib seaborn streamlit shap
```

### 2. Run the Optimization Pipeline
```bash
python run_optimization_pipeline.py
```

### 3. Launch the Web Application
```bash
streamlit run app.py
```

---

## 📄 Documentation & Reports
- **Complete Project Report**: Refer to [`final_project_report.md`](final_project_report.md) and [`Intelligent_Dementia_Detection_Report.docx`](Intelligent_Dementia_Detection_Report.docx).
- **Optimization Results**: Refer to [`optimization_results.md`](optimization_results.md).
