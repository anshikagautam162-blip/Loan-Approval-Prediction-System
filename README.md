# Personal Loan Approval Prediction System

A machine learning project for predicting whether a personal loan application is **Approved** or **Rejected** using applicant financial and demographic information.

## Project Overview

This project develops an end-to-end personal loan approval prediction workflow:

1. Load and inspect the loan dataset.
2. Perform exploratory data analysis (EDA).
3. Investigate missing values, duplicates, feature distributions, and relationships with loan status.
4. Engineer a **loan-to-income ratio** feature.
5. Split the data into training and testing sets using stratification.
6. Preprocess numerical and categorical features.
7. Train and compare multiple classification models.
8. Tune a Decision Tree using 5-fold cross-validation and F1 score for the **Approved** class.
9. Evaluate the selected model using accuracy, precision, recall, F1 score, and a confusion matrix.
10. Perform additional dataset/model sanity checks.
11. Build a Gradio-based prediction interface with input validation.

## Dataset

The notebook uses the **Loan Approval Prediction Dataset** loaded from:

`/kaggle/input/datasets/architsharma01/loan-approval-prediction-dataset/loan_approval_dataset.csv`

The dataset contains **4,269 rows and 13 columns**. The target is `loan_status`, while `loan_id` is excluded from model training.

### Input Features

| Feature | Description |
|---|---|
| `no_of_dependents` | Number of dependents |
| `education` | Graduate / Not Graduate |
| `self_employed` | Yes / No |
| `income_annum` | Annual income |
| `loan_amount` | Requested loan amount |
| `loan_term` | Loan term in years |
| `cibil_score` | CIBIL credit score |
| `residential_assets_value` | Residential asset value |
| `commercial_assets_value` | Commercial asset value |
| `luxury_assets_value` | Luxury asset value |
| `bank_asset_value` | Bank asset value |
| `loan_to_income_ratio` | Loan amount divided by annual income |

### Target

- `Approved`
- `Rejected`

The dataset contains **2,656 Approved** and **1,613 Rejected** applications.

## Data Preparation and EDA

The notebook checks the dataset shape, columns, data types, missing values, duplicate rows, descriptive statistics, target distribution, categorical distributions, and feature relationships with loan status.

No missing values were found, and the notebook found **0 exact duplicate rows**.

The dataset contains 28 records with `residential_assets_value = -100000`. The notebook treats this value as an anomalous value and replaces it with **5,700,000** before model development.

A derived feature, `loan_to_income_ratio`, is also used to represent the requested loan relative to annual income.

## Train/Test Split

The data is divided using:

- **80% training data**
- **20% testing data**
- `random_state = 42`
- `stratify = y`

This produces:

- **3,415 training samples**
- **854 testing samples**

The target column is excluded from the feature matrix, and `loan_id` is also excluded.

## Preprocessing

The notebook uses a `ColumnTransformer`:

### Numerical features

Numerical features are standardized using `StandardScaler`.

### Categorical features

`education` and `self_employed` are encoded using:

- `OneHotEncoder`
- `drop='first'`
- `handle_unknown='ignore'`

The resulting processed feature space contains **12 features**.

## Models Evaluated

The notebook compares several classification approaches:

1. Logistic Regression
2. Decision Tree
3. Pruned Decision Tree
4. Random Forest
5. Tuned Decision Tree

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 93.44% | 97.22% | 92.09% | 94.58% |
| Decision Tree | 99.41% | 99.62% | 99.44% | 99.53% |
| Pruned Decision Tree | **99.65%** | 99.44% | **100.00%** | **99.72%** |
| Random Forest | **99.65%** | 99.44% | **100.00%** | **99.72%** |

The notebook therefore proceeds with Decision Tree tuning because the tree-based models provide very strong performance while the Decision Tree remains relatively easy to interpret.

## Hyperparameter Tuning

The Decision Tree is tuned with `GridSearchCV` using **5-fold cross-validation**.

The search covers:

- `max_depth`: 2–8
- `min_samples_split`: 2, 5, 10
- `min_samples_leaf`: 1, 2, 4

The scoring metric is **F1 score for the `Approved` class**.

### Best Parameters

```text
max_depth = 3
min_samples_leaf = 2
min_samples_split = 2
random_state = 42
```

The best cross-validation F1 score reported by the notebook is approximately **0.99648**.

## Final Tuned Decision Tree

The selected model is:

```text
DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=2,
    min_samples_split=2,
    random_state=42
)
```

### Test Performance

| Metric | Score |
|---|---:|
| Accuracy | **99.65%** |
| Precision | **99.44%** |
| Recall | **100.00%** |
| F1 Score | **99.72%** |

The corresponding confusion matrix is:

```text
[[531   0]
 [  3 320]]
```

The matrix uses the class order:

```text
['Approved', 'Rejected']
```

Therefore, on this test split the model produced:

- 531 correctly classified Approved applications
- 320 correctly classified Rejected applications
- 0 Approved applications classified as Rejected
- 3 Rejected applications classified as Approved

## Threshold Analysis

The notebook also explores different probability thresholds for the earlier Logistic Regression model.

| Threshold | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| 0.50 | 92.12% | 94.73% | 93.41% |
| 0.60 | 95.22% | 93.79% | 94.50% |
| 0.70 | **97.22%** | 92.09% | **94.58%** |
| 0.80 | 98.96% | 89.27% | 93.86% |

A threshold of **0.70** was selected for that probability-based Logistic Regression analysis because it produced the highest F1 score among the tested thresholds.

This threshold analysis is separate from the final tuned Decision Tree selection.

## Model Interpretability

The final tuned Decision Tree uses a small number of decision splits, making it easier to inspect than a large ensemble.

The notebook converts the standardized tree thresholds back into original units. The reported thresholds include:

- **CIBIL score ≤ 549.50**
- **Loan term ≤ 5 years**
- **Loan-to-income ratio ≤ 2.99**
- **Residential assets value ≤ ₹10,50,000 approximately**
- **Loan amount ≤ ₹3,09,50,000 approximately**
- **Luxury assets value ≤ ₹6,00,000 approximately**

These are learned decision thresholds from this dataset and should **not** be interpreted as universal banking rules.

## CIBIL-Only Experiment

To investigate the influence of credit score, a separate Decision Tree using only `cibil_score` was tested.

Its reported accuracy was:

**96.02%**

The notebook also found a strong positive correlation between CIBIL score and the encoded loan status:

**0.7705**

The mean CIBIL scores were:

- Approved: **703.46**
- Rejected: **429.47**

This shows that CIBIL is highly informative in this dataset, while the complete model considers additional features.

## Dataset and Model Sanity Checks

The notebook includes several checks to reduce the risk of misleading evaluation:

- Exact duplicate rows: **0**
- Conflicting groups with identical input features but different loan statuses: **0**
- `loan_status` is excluded from the feature matrix.
- `loan_id` is excluded from the feature matrix.
- Exact feature overlap between training and test sets: **0**

These checks do not establish real-world model validity, but they provide useful sanity checks for this project.

## Prediction Interface

The notebook includes a **Gradio** web interface titled:

**Personal Loan Approval Prediction System**

The interface accepts:

- Number of dependents
- Education
- Self-employment status
- Annual income
- Loan amount
- Loan term
- CIBIL score
- Residential assets
- Commercial assets
- Luxury assets
- Bank assets

The interface calculates the loan-to-income ratio and performs input validation before prediction.

The validation layer prevents unsupported values such as a loan term outside the training-data range or a CIBIL score outside the supported range from being passed directly to the model.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Gradio
- Kaggle Notebook

## Project Workflow

```text
Dataset
   ↓
Data Inspection
   ↓
EDA
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Preprocessing
   ↓
Model Training
   ├── Logistic Regression
   ├── Decision Tree
   ├── Pruned Decision Tree
   └── Random Forest
   ↓
Model Comparison
   ↓
Decision Tree Hyperparameter Tuning
   ↓
Final Tuned Decision Tree
   ↓
Evaluation & Interpretability
   ↓
Input Validation
   ↓
Gradio Prediction Interface
```

## Limitations

This project is based on a single loan-approval dataset and its labels. Therefore:

- The reported performance applies to the dataset and test split used in the notebook.
- The model should not be assumed to have the same performance on real-world banking data.
- Learned tree thresholds are dataset-specific and are not official lending policies.
- Input validation is based on the ranges represented in the training dataset.
- Probability outputs should not be interpreted as guaranteed real-world approval probabilities.
- A real lending system would require additional validation, governance, fairness analysis, monitoring, security, and domain-specific approval policies.

## How to Run

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd <YOUR-REPOSITORY-NAME>
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn gradio
```

### 3. Open the notebook

Open:

```text
loan-approval-prediction-project (1).ipynb
```

in Jupyter Notebook, JupyterLab, Google Colab, or Kaggle.

### 4. Dataset

The notebook currently references the Kaggle input path:

```text
/kaggle/input/datasets/architsharma01/loan-approval-prediction-dataset/loan_approval_dataset.csv
```

If running outside Kaggle, update the CSV path accordingly.

## Project Status

**Completed:** data analysis, preprocessing, model comparison, hyperparameter tuning, evaluation, model interpretation, input validation, and Gradio prediction interface.

---

## Author

**Anshika Gautam**

B.Tech — Artificial Intelligence & Data Science
