# 🏦 Personal Loan Approval Prediction System

A machine learning project for predicting whether a personal loan application is **Approved** or **Rejected** using applicant financial and demographic information.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6.1-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.0-green?logo=pandas)
![Gradio](https://img.shields.io/badge/Gradio-UI-purple)
![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle)

---

## 🔗 Live Demo

👉 **[Try the App Here](https://huggingface.co/spaces/Anshika005/loan-approval-prediction)**

---

## 📌 Project Overview

This project develops an end-to-end personal loan approval prediction workflow:

- Load and inspect the loan dataset
- Perform exploratory data analysis (EDA)
- Investigate missing values, duplicates, feature distributions, and relationships with loan status
- Engineer a `loan_to_income_ratio` feature
- Split data into training and testing sets using stratification
- Preprocess numerical and categorical features
- Train and compare multiple classification models
- Tune a Decision Tree using 5-fold cross-validation and F1 score for the Approved class
- Evaluate the selected model using accuracy, precision, recall, F1 score, and confusion matrix
- Build a Gradio-based prediction interface with input validation

---

## 🔗 Dataset

- **Source:** [Kaggle — Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
- **Records:** 4,269 loan applications
- **Target Distribution:** 2,656 Approved | 1,613 Rejected

| Feature | Description |
|---|---|
| `no_of_dependents` | Number of dependents (0–5) |
| `education` | Graduate / Not Graduate |
| `self_employed` | Yes / No |
| `income_annum` | Annual income (₹) |
| `loan_amount` | Requested loan amount (₹) |
| `loan_term` | Loan term in years |
| `cibil_score` | CIBIL credit score (300–900) |
| `residential_assets_value` | Residential asset value (₹) |
| `commercial_assets_value` | Commercial asset value (₹) |
| `luxury_assets_value` | Luxury asset value (₹) |
| `bank_asset_value` | Bank asset value (₹) |
| `loan_to_income_ratio` | Loan amount ÷ Annual income *(engineered)* |

---

## 🏗️ Project Structure

```
📦 Loan-Approval-Prediction-System
 ┣ 📓 Loan_Approval_Prediction_Project.ipynb   ← Main notebook
 ┗ 📄 README.md
```

---

## 🤖 Models Trained & Compared

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 |
|---|---|---|---|---|
| Logistic Regression | 93.44% | 97.22% | 92.09% | 94.58% |
| Decision Tree (Unpruned) | 99.41% | 99.62% | 99.44% | 99.53% |
| Decision Tree (Pruned, depth=5) | 99.65% | 99.44% | 100.00% | 99.72% |
| Random Forest | 99.65% | 99.44% | 100.00% | 99.72% |
| **Decision Tree (GridSearchCV Tuned)** | **99.65%** | **99.44%** | **100.00%** | **99.72%** |

✅ **Final Model Selected:** Tuned Decision Tree (`max_depth=3, min_samples_leaf=2`)

---

## 📊 Final Model — Confusion Matrix

```
              Predicted
              Approved  Rejected
Actual Approved  531        0
       Rejected    3      320
```

---

## 📈 Key Findings

- **CIBIL Score** is the most important factor — correlation with approval: **0.77**
- Mean CIBIL: Approved = **703.46** | Rejected = **429.47**
- **Loan-to-Income Ratio** is the second most important feature
- CIBIL-only model achieved **96.02% accuracy** — showing credit score alone is highly predictive
- 0 conflicting records and 0 exact duplicate rows found in the dataset

---

## 🔧 Data Preparation

- Stripped whitespace from all column names and string values
- Replaced invalid `residential_assets_value` of -100,000 with median (₹5,700,000) — 28 records affected
- Engineered new feature: `loan_to_income_ratio = loan_amount / income_annum`
- Train/Test Split: **80% / 20%**, stratified, `random_state=42`

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/anshikagautam162-blip/Loan-Approval-Prediction-System
cd Loan-Approval-Prediction-System
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib scikit-learn gradio joblib
```

### 3. Open the notebook
Open `Loan_Approval_Prediction_Project.ipynb` in:
- Kaggle (recommended — dataset is pre-linked)
- Jupyter Notebook / JupyterLab
- Google Colab

### 4. Dataset path
The notebook uses the Kaggle input path:
```
/kaggle/input/loan-approval-prediction-dataset/loan_approval_dataset.csv
```
If running outside Kaggle, update this path to your local CSV file location.

---

## ⚠️ Limitations

- Performance metrics apply to this specific dataset and test split
- Learned decision thresholds are dataset-specific and not official banking rules
- Input validation is based on ranges present in the training data
- Not intended for real-world lending decisions

---

## 👩‍💻 Author

**Anshika Gautam**  
B.Tech — Artificial Intelligence & Data Science  

[GitHub](https://github.com/anshikagautam162-blip) | [Live Demo](https://huggingface.co/spaces/Anshika005/loan-approval-prediction)
