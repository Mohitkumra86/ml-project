# 📡 Telecom Customer Churn Prediction

A complete **Machine Learning project** that predicts whether a telecom customer will churn (leave the service) using classification algorithms. Built with Python, Scikit-learn, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📋 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Objective](#-objective)
3. [Dataset Description](#-dataset-description)
4. [Features Used](#-features-used)
5. [Data Preprocessing](#-data-preprocessing)
6. [Exploratory Data Analysis](#-exploratory-data-analysis)
7. [Classification Algorithms](#-classification-algorithms)
8. [Model Evaluation](#-model-evaluation)
9. [Final Model](#-final-model)
10. [Results](#-results)
11. [Project Structure](#-project-structure)
12. [Installation](#-installation)
13. [How to Run](#-how-to-run)
14. [Expanded Dataset Experiment](#-expanded-dataset-experiment)
15. [Future Improvements](#-future-improvements)

---

## 🔍 Problem Statement

Customer churn is one of the biggest challenges in the telecom industry. When customers leave, the company loses revenue and the cost of acquiring new customers is 5–7× more expensive than retaining existing ones.

**The goal is to identify customers who are likely to churn** so the company can take proactive retention measures such as discounts, loyalty programs, or improved service.

---

## 🎯 Objective

- Build a machine learning classification model that can predict whether a customer will churn or not.
- Train and compare multiple ML algorithms.
- Tune hyperparameters of the best models.
- Deploy the final model as an interactive web application using Streamlit.

---

## 📊 Dataset Description

- **Source:** [Telco Customer Churn Dataset (IBM / Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **File:** `data/Telco-Customer-Churn.csv`
- **Records:** 7,043 customers
- **Features:** 21 columns (20 features + 1 target)
- **Target column:** `Churn` (Yes / No)

| Class | Count | Percentage |
|:------|------:|-----------:|
| No (Retained) | 5,174 | 73.46% |
| Yes (Churned) | 1,869 | 26.54% |

The dataset has **moderate class imbalance** — approximately 1 in 4 customers churned.

---

## 📋 Features Used

### Customer Demographics
| Feature | Type | Values |
|:--------|:-----|:-------|
| `gender` | Categorical | Female, Male |
| `SeniorCitizen` | Binary | 0 (No), 1 (Yes) |
| `Partner` | Categorical | Yes, No |
| `Dependents` | Categorical | Yes, No |
| `tenure` | Numerical | 0–72 months |

### Services Subscribed
| Feature | Type | Values |
|:--------|:-----|:-------|
| `PhoneService` | Categorical | Yes, No |
| `MultipleLines` | Categorical | Yes, No, No phone service |
| `InternetService` | Categorical | DSL, Fiber optic, No |
| `OnlineSecurity` | Categorical | Yes, No, No internet service |
| `OnlineBackup` | Categorical | Yes, No, No internet service |
| `DeviceProtection` | Categorical | Yes, No, No internet service |
| `TechSupport` | Categorical | Yes, No, No internet service |
| `StreamingTV` | Categorical | Yes, No, No internet service |
| `StreamingMovies` | Categorical | Yes, No, No internet service |

### Contract & Billing
| Feature | Type | Values |
|:--------|:-----|:-------|
| `Contract` | Categorical | Month-to-month, One year, Two year |
| `PaperlessBilling` | Categorical | Yes, No |
| `PaymentMethod` | Categorical | Electronic check, Mailed check, Bank transfer, Credit card |
| `MonthlyCharges` | Numerical | 18.25 – 118.75 |
| `TotalCharges` | Numerical | 18.80 – 8684.80 |

---

## 🔧 Data Preprocessing

The preprocessing pipeline ensures clean, consistent data and **avoids data leakage** by fitting transformers only on the training set.

| Step | What We Did | Why |
|:-----|:-----------|:----|
| 1 | Dropped `customerID` | Unique identifier — no predictive value |
| 2 | Converted `TotalCharges` to numeric | 11 rows had blank strings instead of numbers |
| 3 | Encoded `Churn` as 0/1 | No → 0, Yes → 1 for binary classification |
| 4 | Median imputation (numerical) | Handles the 11 missing TotalCharges values |
| 5 | StandardScaler (numerical) | Scales features to zero mean, unit variance |
| 6 | OneHotEncoder (categorical) | Converts categories into binary columns |
| 7 | Stratified train/test split (80/20) | Preserves the class ratio in both sets |

**Final feature count:** 19 raw features → **45 engineered features** after one-hot encoding.

**Script:** `preprocess.py`

---

## 📈 Exploratory Data Analysis

Key insights discovered through EDA (all figures saved in `reports/figures/`):

### 1. Churn by Contract Type
- Month-to-month: **42.7% churn** ← highest risk
- One year: **11.3% churn**
- Two year: **2.8% churn** ← most loyal

### 2. Churn by Internet Service
- Fiber optic customers churn **2× more** than DSL customers
- Customers with no internet have the lowest churn

### 3. Churn vs Tenure
- **New customers (0–12 months)** are the most likely to churn
- Long-tenure customers (48+ months) rarely leave

### 4. Churn vs Monthly Charges
- Higher monthly charges correlate with higher churn
- Churners pay a **median of ~$80/month** vs ~$64 for non-churners

### 5. Key Protective Factors
- Having **Online Security, Tech Support, or Online Backup** significantly reduces churn
- **Automatic payment methods** are associated with lower churn
- **Senior citizens** churn at ~2× the rate of non-seniors

### 6. Top Features (Random Forest Importance)
1. TotalCharges
2. Tenure
3. MonthlyCharges
4. Contract (Month-to-month)
5. OnlineSecurity (No)

**Script:** `eda.py`

---

## 🤖 Classification Algorithms

Five classification algorithms were trained and compared:

| # | Algorithm | Why We Chose It |
|:--|:----------|:---------------|
| 1 | **Logistic Regression** | Simple, interpretable, good baseline for binary classification |
| 2 | **Decision Tree** | Easy to visualize and explain in a viva |
| 3 | **Random Forest** | Ensemble of trees — reduces overfitting |
| 4 | **K-Nearest Neighbors** | Non-parametric, distance-based approach |
| 5 | **Support Vector Machine** | Effective in high-dimensional spaces |

**Script:** `train.py`

---

## 📏 Model Evaluation

### Baseline Model Comparison (before tuning)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|:------|:--------:|:---------:|:------:|:--------:|:-------:|
| Logistic Regression | 0.8055 | 0.6572 | 0.5589 | 0.6040 | 0.8419 |
| K-Nearest Neighbors | 0.7637 | 0.5527 | 0.5749 | 0.5636 | 0.7892 |
| Support Vector Machine | 0.7913 | 0.6418 | 0.4840 | 0.5518 | 0.7905 |
| Random Forest | 0.7807 | 0.6117 | 0.4759 | 0.5353 | 0.8160 |
| Decision Tree | 0.7204 | 0.4745 | 0.4973 | 0.4856 | 0.6489 |

### Why F1 Score Matters More Than Accuracy

For churn prediction, **recall is critical** — missing a churner (false negative) means losing a customer. Plain accuracy can be misleading because of the class imbalance (a model predicting "No Churn" for everyone would score 73.5% accuracy but catch zero churners). The **F1 Score** balances precision and recall.

### Hyperparameter Tuning

The top 2 models (Logistic Regression & Random Forest) were tuned using **RandomizedSearchCV** with 5-fold stratified cross-validation.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|:------|:--------:|:---------:|:------:|:--------:|:-------:|
| LR (baseline) | 0.8055 | 0.6572 | 0.5588 | 0.6040 | 0.8419 |
| LR (tuned) | 0.7615 | 0.5385 | 0.7112 | 0.6129 | 0.8390 |
| RF (baseline) | 0.7807 | 0.6117 | 0.4759 | 0.5353 | 0.8160 |
| **RF (tuned)** | **0.7544** | **0.5245** | **0.8021** | **0.6342** | **0.8395** |

**Script:** `tune_model.py`

---

## 🏆 Final Model

### Tuned Random Forest Classifier

The tuned Random Forest was selected as the **final best model** because:

- **Highest F1 Score (0.634)** — best balance of precision and recall
- **Highest Recall (80.2%)** — catches 300 out of 374 actual churners
- **Strong ROC-AUC (0.840)** — excellent ranking ability

### Best Hyperparameters

```
n_estimators      = 300
max_depth         = 10
min_samples_split = 10
min_samples_leaf  = 2
max_features      = 'sqrt'
class_weight      = {0: 1, 1: 3}
```

The `class_weight = {0: 1, 1: 3}` setting tells the model to penalize missed churners 3× more heavily, which significantly boosts recall.

### Saved Pipeline

The final model is saved as a **complete end-to-end pipeline** that includes both the preprocessor and classifier:

```
models/best_churn_model.pkl
```

This means you can feed **raw customer data** directly into the pipeline — no manual preprocessing needed.

---

## 📊 Results

### Confusion Matrix

|  | Predicted: No Churn | Predicted: Churn |
|:--|:---:|:---:|
| **Actual: No Churn** | 763 (54.2%) | 272 (19.3%) |
| **Actual: Churn** | 74 (5.3%) | 300 (21.3%) |

### Interpretation
- **300 churners correctly identified** (True Positives)
- **74 churners missed** (False Negatives)
- **272 false alarms** (False Positives) — non-churners flagged as churners
- The model catches **80.2% of all churners**, which is valuable for proactive retention

### Risk Levels

| Risk Level | Probability Range | Action |
|:-----------|:-----------------|:-------|
| 🔴 HIGH | ≥ 75% | Immediate retention intervention needed |
| 🟡 MEDIUM | 45% – 75% | Proactive engagement recommended |
| 🟢 LOW | < 45% | Monitor and maintain service quality |

---

## 📁 Project Structure

```
telecom-churn-ml/
│
├── data/
│   ├── Telco-Customer-Churn.csv       # Original dataset (7,043 rows)
│   ├── Telco_customer_churn.xlsx      # Supplementary full dataset
│   └── Telco_customer_churn_status.xlsx # Supplementary status dataset (Satisfaction Score)
│
├── models/                            # Baseline production models (preserved)
│   ├── best_churn_model.pkl           # Baseline pipeline (preprocessor + model)
│   ├── best_churn_model.joblib        # Baseline pipeline in joblib format
│   ├── comparison_table.csv           # Baseline model comparison
│   ├── logistic_regression.joblib     # Individual trained models
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   ├── k-nearest_neighbors.joblib
│   ├── support_vector_machine.joblib
│   └── preprocessor.joblib           # Fitted preprocessor
│
├── models_baseline_experiment/        # Baseline experiment output directory
│   ├── best_churn_model.joblib
│   ├── comparison_table.csv
│   └── preprocessor.joblib
│
├── models_expanded/                   # Expanded experiment output directory
│   ├── best_churn_model_expanded.joblib # Tuned leak-free pipeline (preprocessor + RF)
│   ├── preprocessor_expanded.joblib   # Fitted expanded preprocessor
│   ├── tuned_metrics_expanded.csv     # Test metrics of tuned model
│   └── comparison_table_expanded.csv  # Auxiliary models comparison
│
├── reports/
│   └── figures/                       # Baseline EDA and evaluation figures
│
├── src/                               # Experiment pipelines and expanded application
│   ├── __init__.py
│   ├── split_indices.py               # Shared stratified train/test row indices
│   ├── train_baseline_wrapper.py      # Baseline experiment trainer
│   ├── tune_wrapper.py                # Leak-free cross-validated Pipeline tuner
│   ├── train_wrapper.py               # Expanded auxiliary model trainer
│   ├── run_experiments.py            # CLI orchestrator for baseline vs. expanded
│   ├── app_expanded.py                # Streamlit app with Satisfaction Score & schema helper
│   └── tests/
│       ├── __init__.py
│       └── test_experiments.py       # Isolated pytest test suite (tmp_path safe)
│
├── data_analysis.py                   # Initial data inspection & quality checks
├── preprocess.py                      # Data cleaning & preprocessing pipeline
├── eda.py                             # Exploratory data analysis & visualizations
├── train.py                           # Train & evaluate 5 baseline models
├── tune_model.py                      # Hyperparameter tuning & final model selection
├── predict.py                         # Command-line prediction script
├── app.py                             # Streamlit web application (baseline)
├── README.md                          # This file
└── requirements.txt                   # Python dependencies
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Step 1: Clone or download the project

```bash
cd telecom-churn-ml
```

### Step 2: Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### Step 3: Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit
```

---

## 🚀 How to Run

### 1. Data Analysis (inspect the dataset)

```bash
python data_analysis.py
```

This prints dataset shape, column types, missing values, duplicates, target distribution, and data-quality issues.

### 2. Exploratory Data Analysis (generate graphs)

```bash
python eda.py
```

Saves 10 visualizations to `reports/figures/`.

### 3. Train Baseline Models

```bash
python train.py
```

Trains 5 classifiers, prints evaluation metrics, and saves models to `models/`.

### 4. Hyperparameter Tuning

```bash
python tune_model.py
```

Tunes Logistic Regression and Random Forest, compares with baselines, saves the final pipeline to `models/best_churn_model.pkl`.

### 5. Predict from Command Line

```bash
python predict.py
```

Runs predictions on 4 built-in sample customers and prints churn probability, prediction, and risk level.

### 6. Launch the Baseline Streamlit Web App

```bash
streamlit run app.py
```

Opens a browser at `http://localhost:8501` where you can enter customer details and get real-time predictions from the baseline production model.

### 7. Run Unified Experiment Comparison (Baseline vs. Expanded)

```bash
python -m src.run_experiments
```

Loads the raw CSV once, generates shared stratified train/test row indices, trains baseline models in `models_baseline_experiment/`, tunes the leak-free expanded pipeline in `models_expanded/`, and prints a side-by-side performance table.

Custom output directories can be specified:
```bash
python -m src.run_experiments --baseline_dir <path> --expanded_dir <path>
```

### 8. Launch the Expanded Streamlit Web App

```bash
streamlit run src/app_expanded.py
```

Launches the experimental dashboard allowing users to evaluate churn risk alongside customer `Satisfaction Score` (1–5).

### 9. Run the Test Suite

```bash
pytest src/tests/ -v
```

Executes 9 automated tests covering split reproducibility, leak-free pipeline tuning, input schema synchronization, and zero-leakage assertions.

---

## 🧪 Expanded Dataset Experiment

### Overview
The expanded dataset experiment incorporates supplementary customer status information from `data/Telco_customer_churn_status.xlsx`, merged with `data/Telco-Customer-Churn.csv` on `customerID = Customer ID`.

- **Targeted Feature Added:** `Satisfaction Score` (integer scale 1–5).
- **Target & Leakage Columns Excluded:** `customerID`, `Customer ID`, `Status ID`, `Count`, `Quarter`, `Churn Label`, `Churn Value`, `Churn Score`, `CLTV`, `Churn Reason`, `Churn Category`, and `Customer Status`.
- **Shared Stratified Partition:** Both baseline and expanded experiments evaluate on identical row splits generated via `compute_global_indices()` (`test_size=0.2, random_state=42`).
- **Leak-Free Cross-Validation:** Hyperparameter tuning evaluates a complete `scikit-learn` `Pipeline(steps=[('preprocessor', ColumnTransformer), ('classifier', RandomForestClassifier)])`. Preprocessing transformations are fit strictly within each fold during 5-fold `StratifiedKFold` cross-validation on raw training data (`X_train_raw`). The held-out test set (`X_test_raw`) is evaluated only after tuning finishes.

### Comparative Results Summary

| Pipeline | Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | Logistic Regression | 80.55% | 65.72% | 55.88% | 0.6040 | 0.8419 |
| **Baseline (Selected)** | Tuned Random Forest | 75.44% | 52.45% | 80.21% | 0.6342 | 0.8395 |
| **Expanded (Experimental)** | Tuned Random Forest (+Satisfaction Score) | **94.46%** | **88.95%** | **90.37%** | **0.8966** | **0.9794** |

### ⚠️ Retrospective Target Leakage Caveat
`Satisfaction Score` exhibits exceptionally strong predictive power, driving the model's F1 Score from 0.63 to ~0.90. However, in telecommunications operations, satisfaction surveys are frequently collected at the end of a billing quarter or during churn exit interviews. When a score reflects post-dissatisfaction sentiments collected concurrently with account closure, it introduces **retrospective target leakage**.

For this reason:
- The baseline model in `models/` remains the **primary production benchmark**.
- The expanded model in `models_expanded/` is classified as an **experimental investigation**.
- All user interfaces and scripts prominently display leakage warnings.

---

## 🔮 Future Improvements

| Improvement | Description |
|:-----------|:-----------|
| **SMOTE oversampling** | Generate synthetic minority samples to better handle class imbalance |
| **Gradient Boosting models** | Try XGBoost, LightGBM, or CatBoost for potentially better performance |
| **Feature engineering** | Create derived features like `avg_monthly_charge = TotalCharges / tenure` |
| **Threshold optimization** | Tune the classification threshold (currently 0.5) to maximize recall |
| **SHAP explainability** | Integrated into the Observatory dashboard for local customer-level attributions |
| **Cross-validation reporting** | Report mean ± std across CV folds for more robust evaluation |
| **Database integration** | Connect to a database to store and retrieve customer records |
| **Batch prediction** | Allow uploading a CSV of multiple customers for bulk prediction |
| **Model monitoring** | Track model drift and retrain on new data periodically |

---

## 🛠️ Technologies Used

| Technology | Purpose |
|:-----------|:--------|
| **Python** | Programming language |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computing |
| **Scikit-learn** | ML algorithms, preprocessing, evaluation |
| **Matplotlib** | Data visualization |
| **Seaborn** | Statistical plots |
| **Streamlit** | Web application framework |
| **Joblib / Pickle** | Model serialization |

---

## 📚 References

- [Telco Customer Churn Dataset – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

<p align="center">
  <b>Built as a College ML Project</b><br>
  Telecom Customer Churn Prediction using Machine Learning
</p>
