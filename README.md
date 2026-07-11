# 💳 Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-green.svg)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue.svg)
![GitHub Actions](https://img.shields.io/badge/CI-GitHub%20Actions-success.svg)

An end-to-end **Machine Learning pipeline** for detecting fraudulent credit card transactions using multiple classification algorithms, automated hyperparameter tuning, MLflow experiment tracking, Docker, GitHub Actions CI/CD, and an interactive Streamlit dashboard.

---

# 📑 Table of Contents

- Project Overview
- Features
- Tech Stack
- Project Structure
- Machine Learning Pipeline
- Models Used
- Performance
- Installation
- Usage
- Streamlit Dashboard
- MLflow Tracking
- Docker
- GitHub Actions
- Testing
- Future Improvements
- Author

---

# 📌 Project Overview

Credit card fraud causes significant financial losses every year. Since fraudulent transactions represent only a very small percentage of all transactions, fraud detection becomes a highly imbalanced binary classification problem.

This project provides a production-inspired machine learning pipeline that automatically preprocesses data, trains multiple models, performs hyperparameter tuning, selects the best-performing model, evaluates it using business-relevant metrics, and serves predictions through an interactive Streamlit application.

---

# ✨ Features

- End-to-End Machine Learning Pipeline
- Automated Data Preprocessing
- Duplicate & Missing Value Handling
- Robust Feature Scaling
- Stratified Train-Test Split
- SMOTE for Imbalanced Data
- Hyperparameter Tuning using RandomizedSearchCV
- Multiple Model Comparison
- Automatic Best Model Selection
- MLflow Experiment Tracking
- MLflow Model Registry
- Threshold Optimization
- Batch Prediction
- Single Transaction Prediction
- Interactive Streamlit Dashboard
- Docker Support
- GitHub Actions CI
- Unit Testing

---

# 🛠 Tech Stack

### Programming

- Python

### Machine Learning

- Scikit-Learn
- XGBoost
- Imbalanced-Learn

### Data Processing

- Pandas
- NumPy

### Visualization

- Plotly
- Matplotlib

### Deployment

- Streamlit
- Docker

### Experiment Tracking

- MLflow

### Testing

- Pytest

### CI/CD

- GitHub Actions

---

# 📂 Project Structure

```text
credit_card_fraud_detection/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   └── creditcard.csv
│
├── models/
│   ├── best_fraud_model.joblib
│   └── scaler.joblib
│
├── notebooks/
│
├── outputs/
│   ├── metrics.json
│   └── roc_pr_curves.png
│
├── src/
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── __init__.py
│
├── tests/
│   └── test_pipeline.py
│
├── app.py
├── main.py
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── README.md
```

---

# 🔄 Machine Learning Pipeline

```text
Raw Dataset
      │
      ▼
Data Validation
      │
      ▼
Data Cleaning
      │
      ▼
Feature Scaling
(RobustScaler)
      │
      ▼
Train/Test Split
(Stratified)
      │
      ▼
SMOTE
(Class Balancing)
      │
      ▼
Hyperparameter Tuning
(RandomizedSearchCV)
      │
      ▼
Model Comparison
      │
      ▼
Best Model Selection
      │
      ▼
Model Saving
      │
      ▼
Model Evaluation
      │
      ▼
MLflow Logging
      │
      ▼
Streamlit Dashboard
```

---

# 📊 Data Preprocessing

The preprocessing pipeline performs the following operations:

- Dataset validation
- Duplicate removal
- Missing value handling
- Feature-target separation
- Stratified train-test split
- Robust scaling for **Time** and **Amount**
- Scaler serialization using Joblib

---

# 🤖 Models Used

The pipeline automatically trains and compares multiple machine learning models.

| Model | Hyperparameter Tuning | Class Imbalance Handling |
|--------|-----------------------|--------------------------|
| Logistic Regression | ✅ | SMOTE |
| Random Forest | ✅ | SMOTE |
| XGBoost | ✅ | SMOTE |

The best-performing model is automatically selected and saved.

---

# 📈 Model Performance

## Best Model

**XGBoost Classifier**

| Metric | Score |
|--------|--------:|
| ROC-AUC | **0.9726** |
| PR-AUC | **0.7916** |
| Accuracy | **99.70%** |
| Fraud Precision | **33.91%** |
| Fraud Recall | **83.16%** |
| Fraud F1-Score | **48.17%** |

### Confusion Matrix

| | Predicted Genuine | Predicted Fraud |
|---|---:|---:|
| Actual Genuine | 56,497 | 154 |
| Actual Fraud | 16 | 79 |

> **Note:** The dataset is highly imbalanced. Therefore, ROC-AUC, PR-AUC, Precision, Recall, and F1-score provide a more meaningful evaluation than accuracy alone.

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
```

Move into the project directory

```bash
cd credit-card-fraud-detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Train the Model

```bash
python main.py
```

This pipeline automatically performs:

- Data preprocessing
- Feature scaling
- Model training
- Hyperparameter tuning
- Model comparison
- Best model selection
- Evaluation
- MLflow logging

---

# 💻 Run the Streamlit Dashboard

```bash
streamlit run app.py
```

---

# 🖥 Dashboard Features

### Batch Prediction

- Upload CSV file
- Fraud probability estimation
- Fraud detection
- Download prediction results
- Fraud distribution visualization
- Confusion matrix (if labels are available)

---

### Single Transaction Prediction

Manually enter transaction information including:

- Time
- Amount
- PCA Features (V1–V28)

The dashboard predicts:

- Fraud Probability
- Fraud / Genuine Decision

---

### Model Performance Dashboard

Displays:

- ROC-AUC
- PR-AUC
- Classification Report
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

---

# 📊 MLflow Experiment Tracking

The project uses MLflow for:

- Experiment Tracking
- Hyperparameter Logging
- Metric Logging
- Model Versioning
- Model Registry

Launch MLflow using

```bash
mlflow ui
```

---

# 🐳 Docker

Build Docker image

```bash
docker build -t fraud-detection .
```

Run Docker container

```bash
docker run -p 8501:8501 fraud-detection
```

Or use Docker Compose

```bash
docker-compose up
```

---

# ⚙ GitHub Actions CI

The project includes an automated CI pipeline that:

- Installs project dependencies
- Executes unit tests
- Performs Python syntax validation
- Builds the Docker image

This ensures that every push and pull request is automatically verified before integration.

---

# 🧪 Running Tests

Execute all tests

```bash
pytest tests/ -v
```

---

# 📦 Requirements

Main libraries used:

- pandas
- numpy
- scikit-learn
- xgboost
- imbalanced-learn
- mlflow
- streamlit
- plotly
- matplotlib
- pytest
- joblib

---

# 🚀 Future Improvements

- Explainable AI using SHAP
- Drift Detection
- Model Monitoring
- REST API using FastAPI
- Real-time Fraud Detection
- Cloud Deployment (AWS / Azure / GCP)
- User Authentication
- Continuous Model Retraining

---

# 👨‍💻 Author

**Sajid Ali**

B.Tech CSE (Data Science)

Jamia Millia Islamia

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
