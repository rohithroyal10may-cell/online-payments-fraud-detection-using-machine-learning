# Online Payments Fraud Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python) ![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey?logo=flask) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn) ![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-brightgreen) ![License](https://img.shields.io/badge/License-MIT-yellow)

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Features](#features)
- [Project Workflow](#project-workflow)
  - [1. Data Collection](#1-data-collection)
  - [2. Exploratory Data Analysis](#2-exploratory-data-analysis)
  - [3. Data Preprocessing](#3-data-preprocessing)
  - [4. Model Building](#4-model-building)
  - [5. Model Comparison](#5-model-comparison)
  - [6. Model Deployment](#6-model-deployment)
- [Machine Learning Models](#machine-learning-models)
- [Web Application](#web-application)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [License](#license)

---

## Overview

Online Payments Fraud Detection is an end-to-end machine learning project that detects fraudulent transactions in digital payment systems. The rapid growth of online credit/debit card usage has led to a parallel rise in financial fraud. This project addresses that challenge by training multiple classification models on a real-world transaction dataset, selecting the best-performing model, and serving it through an interactive Flask web application that allows users to classify any transaction as **Fraud** or **Not Fraud** in real time.

---

## Problem Statement

The surge in online payments has increased exposure to fraudulent activities such as unauthorised transfers and cash-outs. Traditional rule-based detection systems struggle with the volume and complexity of modern transactions. This project leverages supervised machine learning to learn patterns from historical transaction data and predict whether a new transaction is fraudulent, enabling faster and more accurate fraud prevention.

---

## Dataset

- **Source:** [Kaggle – Online Payments Fraud Detection Dataset](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset)
- **Original size:** ~6 million transactions
- **Balanced size used for training:** 16,426 transactions (8,213 legitimate + 8,213 fraudulent)

The original dataset is highly imbalanced (far more legitimate transactions than fraudulent ones). To avoid model bias, a balanced subset was created by random-sampling legitimate transactions to match the number of fraudulent ones. The balanced dataset is saved as `balanced_dataset.csv`.

---

## Features

| Feature | Description |
|---|---|
| `step` | Unit of time (1 step = 1 hour). Range: 1–743 |
| `type` | Transaction type (encoded as integer — see below) |
| `amount` | Amount of the transaction in local currency |
| `oldbalanceOrg` | Sender's account balance before the transaction |
| `newbalanceOrig` | Sender's account balance after the transaction |
| `oldbalanceDest` | Recipient's account balance before the transaction |
| `newbalanceDest` | Recipient's account balance after the transaction |
| `isFraud` | **Target variable** — 1 = Fraudulent, 0 = Legitimate |

### Transaction Type Encoding

| Label | Encoded Value |
|---|---|
| CASH_IN | 0 |
| CASH_OUT | 1 |
| DEBIT | 2 |
| PAYMENT | 3 |
| TRANSFER | 4 |

> **Note:** The `isFlaggedFraud`, `nameOrig`, and `nameDest` columns were dropped during preprocessing as they did not contribute meaningful signal to the models.

---

## Project Workflow

### 1. Data Collection

The raw dataset is sourced from Kaggle and loaded with `pandas`. The `isFlaggedFraud` column is removed immediately as it contains only one unique value.

### 2. Exploratory Data Analysis

Both univariate and bivariate analyses were performed to understand the data distribution:

- **Univariate analysis:** histograms, box plots, and count plots for individual features (`step`, `amount`, `type`, `oldbalanceOrg`, etc.)
- **Bivariate analysis:** joint plots, count plots, box plots, and violin plots comparing features against the target label `isFraud`
- **Correlation heatmap:** to identify relationships between numerical features

Key finding: Fraudulent transactions are predominantly of type `CASH_OUT` and `TRANSFER`.

### 3. Data Preprocessing

| Step | Detail |
|---|---|
| Class balancing | Random under-sampling of the majority class (legitimate transactions) to produce a 50/50 split |
| Dropped columns | `nameOrig`, `nameDest` (identifiers, not predictive) |
| Null-value check | No missing values found |
| Outlier handling | IQR method applied to `amount`, `oldbalanceOrg`, and `newbalanceOrig`; outliers replaced with the column mean |
| Label encoding | `type` (categorical string) encoded to integers using `LabelEncoder` |
| Train/test split | 80% training, 20% testing (`random_state=0`) |

### 4. Model Building

Five classification algorithms were trained and evaluated on the preprocessed dataset.

### 5. Model Comparison

After comparing training and testing accuracy across all five models, **XGBoost Classifier** achieved the highest test accuracy without overfitting and was selected as the final model.

The trained model is serialised with `pickle` and saved as `payments.pkl`.

### 6. Model Deployment

The saved model is loaded by a Flask web application (`app.py`) that exposes a prediction endpoint. Users input transaction details via a web form, and the app returns a real-time classification result.

---

## Machine Learning Models

| # | Algorithm | Description |
|---|---|---|
| 1 | **Random Forest Classifier** | Ensemble of decision trees trained on random data/feature subsets; reduces overfitting via aggregation |
| 2 | **Decision Tree Classifier** | Tree-structured model that recursively splits data based on feature thresholds; highly interpretable |
| 3 | **Extra Trees Classifier** | Like Random Forest but uses fully random split points, improving computational efficiency |
| 4 | **Support Vector Machine (SVC)** | Finds the optimal separating hyperplane that maximises the margin between classes |
| 5 | **XGBoost Classifier** ✅ | Gradient-boosting ensemble that iteratively combines weak learners; selected as the best model |

Evaluation metrics used: **accuracy score**, **F1 score**, **classification report**, and **confusion matrix**.

---

## Web Application

The project includes a Flask web application with three pages:

| Page | Route | Description |
|---|---|---|
| Home | `/` | Project introduction and description |
| Predict | `/predict` | Form to enter transaction details and get a prediction |
| Result | `/predict` (POST) | Displays whether the transaction is **Fraud** or **Not Fraud** |

### Input Fields (Prediction Form)

- **Step** — Time step of the transaction
- **Type** — Transaction type (use the encoded integer: 0–4)
- **Amount** — Transaction amount
- **Old Balance Org** — Sender's balance before transaction
- **New Balance Orig** — Sender's balance after transaction
- **Old Balance Dest** — Recipient's balance before transaction
- **New Balance Dest** — Recipient's balance after transaction

---

## Project Structure

```
Online-Payments-Fraud-Detection-using-Machine-Learning/
│
├── code files/
│   ├── app.py                        # Flask web application
│   ├── fraud detection.ipynb         # Jupyter notebook (EDA, preprocessing, model training)
│   ├── fraud detection dataset.csv   # Raw dataset from Kaggle (not committed; must be downloaded)
│   ├── balanced_dataset.csv          # Balanced dataset used for model training
│   ├── payments.pkl                  # Serialised XGBoost model
│   ├── static/
│   │   ├── style.css                 # Application stylesheet
│   │   ├── background.jpg            # Background image
│   │   ├── logo.png                  # Application logo / favicon
│   │   └── Techno.ttf                # Custom font
│   └── templates/
│       ├── home.html                 # Landing page template
│       ├── predict.html              # Prediction form template
│       └── submit.html               # Result display template
│
├── Project Documentation/
│   ├── 1. Empath Map for Online Payments Fraud Final.pdf
│   ├── 2. Brainstorming 1 Final.pdf
│   ├── 3. Brainstorming 2 Final.pdf
│   ├── 4. Proposed Solution.pdf
│   ├── 5. Solution Architect.pdf
│   ├── 6. Data Flow Diagram and User Stories.pdf
│   ├── 7. Technology Stack.pdf
│   ├── 8. Project Planning Details.pdf
│   ├── 9. Project Development Phase.pdf
│   └── 10. Solution Performance.pdf
│
├── LICENSE
└── README.md
```

---

## Technology Stack

| Category | Technology |
|---|---|
| Language | Python 3.x |
| Data manipulation | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine learning | scikit-learn, XGBoost |
| Model serialisation | pickle / joblib |
| Web framework | Flask |
| Front-end | HTML5, CSS3 |
| Statistical analysis | SciPy |

---

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip

### Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/Saket8538/Online-Payments-Fraud-Detection-using-Machine-Learning.git
   cd Online-Payments-Fraud-Detection-using-Machine-Learning
   ```

2. **Install dependencies**

   ```bash
   pip install flask pandas numpy scikit-learn xgboost joblib scipy matplotlib seaborn
   ```

3. **Navigate to the application directory**

   ```bash
   cd "code files"
   ```

4. **(Optional) Retrain the model**

   Open `fraud detection.ipynb` in Jupyter Notebook or JupyterLab and run all cells. This will generate a fresh `payments.pkl` file. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset) and save it as `fraud detection dataset.csv` in the `code files/` directory before running the notebook.

5. **Run the Flask application**

   ```bash
   python app.py
   ```

6. **Open in browser**

   Navigate to `http://127.0.0.1:5000/` in your web browser.

---

## Usage

1. Open the web application at `http://127.0.0.1:5000/`.
2. Click **Predict** in the navigation bar.
3. Fill in the transaction details:
   - Enter the transaction type as its encoded integer (e.g., `1` for CASH_OUT).
   - Enter all balance and amount values as numbers.
4. Click **Submit**.
5. The application displays whether the transaction is classified as **Fraud** or **Not Fraud**.

### Example Input

| Field | Example Value |
|---|---|
| Step | 1 |
| Type | 4 (TRANSFER) |
| Amount | 181.00 |
| Old Balance Org | 181.00 |
| New Balance Orig | 0.00 |
| Old Balance Dest | 0.00 |
| New Balance Dest | 0.00 |

---

## License

This project is licensed under the [MIT License](LICENSE).
