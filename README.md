# Credit Card Fraud Detection & Risk Analysis System

Machine Learning-Based Fraud Detection and Transaction Risk Analysis — an end-to-end, portfolio-grade ML system built from the Kaggle **Credit Card Fraud Detection (2023)** dataset.

![Fraud vs Legitimate](visualizations/class_distribution.png)

## Project Overview

This project takes a raw 568,630-row credit-card transaction dataset and turns it into a working fraud-detection product: a data pipeline, five compared classification models, a tuned production threshold, a Flask web application ("CreditGuard AI") for real-time predictions, a standalone analytics dashboard, and a full written report — all generated from and validated against the actual dataset, with no fabricated metrics.

## Business Problem

Financial institutions cannot manually review every transaction. Automated fraud detection must catch as much fraud as possible (recall) while minimizing false alarms on legitimate customers (precision) — accuracy alone is a misleading metric when the two error types carry very different real-world costs.

## Objectives

- Programmatically inspect the dataset with no assumptions about its structure
- Clean and preprocess data with a reusable, leakage-free pipeline
- Perform EDA that answers concrete analytical questions
- Engineer only features genuinely supported by the data
- Train and fairly compare multiple classifiers using fraud-appropriate metrics
- Select a production model and decision threshold, justified with real results
- Ship a working prediction web app and analytics dashboard
- Document everything in a professional report

## Key Features

- **Automatic target/ID detection** — no hard-coded column names
- **5 models compared**: Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting, XGBoost
- **Threshold optimization** — sweeps thresholds and selects by F1, not the default 0.5
- **Reusable preprocessing pipeline** — identical scaling at training and inference time
- **Flask web app** with a live transaction-risk form and real dataset example presets
- **Standalone HTML dashboard** — opens directly in a browser, no server required
- **Full Word report** — auto-generated from real results via `python-docx`
- **Automated tests** for data loading, preprocessing, and prediction

## Dataset

- **Source**: Kaggle "Credit Card Fraud Detection Dataset 2023"
- **File**: `data/raw/creditcard_2023.csv`
- **Rows**: 568,630 transactions
- **Columns**: `id`, `V1`–`V28` (anonymized PCA components), `Amount`, `Class` (target)
- **Class balance**: 50.00% legitimate / 50.00% fraud (this 2023 version is synthetically balanced, unlike the original 2013 ULB dataset)
- No missing values, duplicate rows, or infinite values were found on inspection.

## Technologies Used

Python · pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Seaborn · Flask · joblib · python-docx · pytest

## Machine Learning Approach

1. **Data loading & inspection** (`src/data_loader.py`) — auto-detects the target column, ID columns, dtypes, and data-quality issues
2. **Cleaning** (`src/preprocessing.py`) — deduplication, missing/infinite-value checks, ID exclusion
3. **Feature engineering** (`src/feature_engineering.py`) — adds `log_amount` (log1p of `Amount`); no synthetic features invented
4. **Modeling** (`src/train.py`) — stratified 80/20 split, `StandardScaler` pipeline, 5 models trained and compared
5. **Threshold tuning** — sweeps 0.05–0.95, selects by F1-score, stores the result in `models/model_metadata.json`
6. **Evaluation** (`src/evaluation.py`) — accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix

## Data Pipeline

```
data/raw/creditcard_2023.csv
        │
        ▼
src/data_loader.py        → target/ID detection, quality summary
        │
        ▼
src/preprocessing.py      → cleaning, train/test split, scaling pipeline
        │
        ▼
src/feature_engineering.py→ log_amount
        │
        ▼
src/train.py              → trains 5 models, selects best, tunes threshold
        │
        ▼
models/best_model.pkl, preprocessing_pipeline.pkl, model_metadata.json
        │
        ▼
app/app.py (Flask)  +  dashboard/dashboard.html  +  reports/Credit_Card_Fraud_Report.docx
```

## EDA

Highlights (see `visualizations/` for all charts):

- Fraud vs legitimate transactions are perfectly balanced (284,315 each)
- Average transaction amount is nearly identical across classes (~$12,026 legitimate vs ~$12,058 fraud) — amount alone is a weak fraud signal
- `V14`, `V4`, and `V12` are the features most correlated with the fraud label

## Feature Engineering

Only `log_amount` (a log1p transform of `Amount`) was added. The `V1`–`V28` columns are already PCA outputs with no further engineering possible without inventing data, and there is no timestamp or account ID to build time/velocity features from.

## Models

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC | Train Time |
|---|---|---|---|---|---|---|---|
| **XGBoost** ⭐ | 0.9984 | 0.9975 | 0.9993 | **0.9984** | 0.9999 | 0.9999 | 6.16s |
| HistGradientBoosting | 0.9977 | 0.9969 | 0.9984 | 0.9977 | 0.9999 | 0.9998 | 7.19s |
| Random Forest | 0.9863 | 0.9979 | 0.9747 | 0.9861 | 0.9993 | 0.9993 | 60.12s |
| Decision Tree | 0.9862 | 0.9857 | 0.9868 | 0.9862 | 0.9965 | 0.9959 | 19.56s |
| Logistic Regression | 0.9649 | 0.9771 | 0.9521 | 0.9644 | 0.9935 | 0.9944 | 0.95s |

*(Table generated from `reports/model_results.csv` — see it for the full-precision values.)*

## Evaluation Metrics

**XGBoost** was selected — not for having the highest accuracy, but the best F1-score, balancing precision and recall. At the tuned production threshold of **0.55**:

- Precision: **0.9981** · Recall: **0.9989** · F1: **0.9985** · ROC-AUC: **0.9999**
- Confusion matrix (113,726 test transactions): 56,755 TN · 108 FP · 65 FN · 56,798 TP

## Results

- Best model: **XGBoost**
- Production threshold: **0.55** (tuned, not the 0.5 default)
- Top predictive feature: **V14** (~61% of total feature importance)
- Full breakdown in `reports/Credit_Card_Fraud_Report.docx` and `reports/model_results.csv`

## Dashboard

`dashboard/dashboard.html` is a standalone, single-page analytics dashboard (KPIs, charts, business insights, risks, recommendations) that opens directly in any browser — no Flask server required.

## Web Application

`app/app.py` runs **CreditGuard AI**, a Flask app with a fintech-styled UI where you can:

- Enter a transaction's `Amount` and `V1`–`V28` values (or load a real dataset example)
- Get a LEGITIMATE/FRAUD prediction, a risk score %, and a LOW/MEDIUM/HIGH risk tier
- See the model name and production threshold used

The app loads the saved model and pipeline — it never retrains on startup.

## Project Structure

```
credit-card-fraud-detection/
├── data/
│   ├── raw/creditcard_2023.csv
│   └── processed/cleaned_dataset.csv
├── notebooks/                  # numbered pipeline stage scripts
│   ├── 01_data_exploration.py
│   ├── 02_data_preprocessing.py
│   ├── 03_feature_engineering.py
│   ├── 04_model_training.py
│   └── 05_model_evaluation.py
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluation.py
│   └── generate_report.py
├── models/
│   ├── best_model.pkl
│   ├── preprocessing_pipeline.pkl
│   └── model_metadata.json
├── visualizations/              # all PNG charts
├── dashboard/dashboard.html
├── app/
│   ├── app.py
│   ├── templates/index.html
│   └── static/{style.css, script.js, sample_transactions.json}
├── reports/
│   ├── Credit_Card_Fraud_Report.docx
│   ├── model_results.csv
│   ├── threshold_analysis.csv
│   ├── feature_importance.csv
│   ├── data_quality_summary.json
│   └── eda_findings.json
├── tests/
├── README.md
├── requirements.txt
├── .gitignore
└── run_project.bat
```

## Installation

```bash
git clone <your-repo-url>
cd credit-card-fraud-detection
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

> **Dataset**: place `creditcard_2023.csv` (Kaggle's "Credit Card Fraud Detection Dataset 2023") into `data/raw/creditcard_2023.csv` before running. It is not bundled in this repository/ZIP to keep the download small, since all trained artifacts (`models/`), visualizations, the report, and the dashboard were already generated from it and are included as-is.

## How to Run

**Option 1 — one command (Windows):**
```bash
run_project.bat
```
This installs dependencies, runs the pipeline if no trained model exists yet, and starts the Flask app at `http://127.0.0.1:5000`.

**Option 2 — step by step:**
```bash
python -m src.data_loader          # data-quality summary
python -m src.preprocessing        # cleaning
python notebooks/01_data_exploration.py   # EDA visualizations
python -m src.train                # train, compare, select, save model
python -m src.generate_report      # build the Word report
python app/app.py                  # launch the web app
```

Open `dashboard/dashboard.html` directly in a browser at any time — no server needed.

## How to Use

1. Open `http://127.0.0.1:5000` after starting the app
2. Enter a transaction amount and the `V1`–`V28` signals, or click "Load legitimate/fraud example" to use a real transaction from the dataset
3. Click **Analyze transaction** to see the prediction, risk score, and risk tier

## Screenshots

See `visualizations/` for all generated charts, including:
- `class_distribution.png`, `fraud_amount_distribution.png`
- `correlation_heatmap.png`, `feature_target_correlation.png`
- `model_comparison.png`, `confusion_matrix.png`, `roc_curve.png`, `precision_recall_curve.png`
- `threshold_optimization.png`, `feature_importance.png`

## Key Business Insights

- The dataset is synthetically balanced (50/50); real production traffic will be far more imbalanced and the threshold should be re-validated before deployment
- Transaction amount alone barely differs between fraud and legitimate transactions — it is not a reliable standalone signal here
- A small number of PCA components (led by `V14`) drive most of the model's decisions
- XGBoost's F1-score is ~3.4 points higher than Logistic Regression's, showing real value from gradient boosting on this problem

## Limitations

- PCA-anonymized features limit plain-language explainability
- No timestamp or account ID prevents time/velocity-based feature engineering
- Dataset's 50/50 balance does not reflect typical real-world fraud rates
- Even the best model still has some false positives and false negatives

## Future Enhancements

- Real transaction metadata (timestamps, merchant, geography) for richer features
- Systematic hyperparameter search with cross-validation
- SHAP-based per-prediction explainability
- Evaluation against a realistically imbalanced holdout set
- Analyst-feedback loop for periodic retraining

## Author

Built as a data science / ML engineering portfolio project.
