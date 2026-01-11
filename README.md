# Stellar Classification using Machine Learning

## 1. Problem Statement
Stellar classification is a fundamental task in astronomy that involves identifying celestial objects as **stars, galaxies, or quasars** based on their observed spectral and positional characteristics.  
This project applies multiple machine learning classification algorithms to classify astronomical objects using data from the Sloan Digital Sky Survey (SDSS). The goal is to compare the performance of different models and deploy them in an interactive web application.

---

## 2. Dataset Description
The dataset used is the **Stellar Classification Dataset (SDSS17)** obtained from Kaggle.  
It consists of **100,000 observations** with **17 original features** and one target class.

### Target Variable
- **class**: Object class (`STAR`, `GALAXY`, `QSO`)

### Selected Features (12)
After removing unique identifiers and non-informative columns, the following features were used:
- alpha
- delta
- u, g, r, i, z
- run_ID
- rerun_ID
- cam_col
- field_ID
- redshift

Identifier columns such as `obj_ID`, `spec_obj_ID`, `plate`, `MJD`, and `fiber_ID` were removed as they do not contribute predictive information.

---

## 3. Models Used and Evaluation Metrics
All models were trained on the same dataset and evaluated using the following metrics:
- Accuracy
- AUC (One-vs-Rest, macro average)
- Precision (macro)
- Recall (macro)
- F1 Score (macro)
- Matthews Correlation Coefficient (MCC)

### Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.9411 | 0.9835 | 0.9386 | 0.9297 | 0.9341 | 0.8951 |
| Decision Tree | 0.9659 | 0.9703 | 0.9600 | 0.9613 | 0.9607 | 0.9395 |
| kNN | 0.9165 | 0.9624 | 0.9229 | 0.8923 | 0.9063 | 0.8502 |
| Naive Bayes | 0.7437 | 0.9384 | 0.8064 | 0.6474 | 0.6138 | 0.5384 |
| Random Forest | 0.9793 | 0.9946 | 0.9797 | 0.9721 | 0.9758 | 0.9631 |
| XGBoost | 0.9758 | 0.9955 | 0.9754 | 0.9689 | 0.9720 | 0.9570 |

---

## 4. Observations on Model Performance

| Model | Observation |
|------|-------------|
| Logistic Regression | Provided a strong baseline with high AUC, indicating good linear separability of features. |
| Decision Tree | Achieved higher accuracy but may overfit due to deep splits. |
| kNN | Showed good performance but lower recall due to sensitivity to distance metrics. |
| Naive Bayes | Performed weakest as feature independence assumptions are violated, though AUC remained high. |
| Random Forest | Achieved the best overall performance with highest accuracy and MCC due to ensemble learning. |
| XGBoost | Delivered performance close to Random Forest and effectively captured non-linear feature interactions. |

---

## 5. Streamlit Web Application
An interactive Streamlit application was developed and deployed using **Streamlit Community Cloud**.

### Features:
- CSV dataset upload (test data)
- Model selection dropdown
- Prediction results display
- Evaluation metrics (when true labels are available)
- Confusion matrix visualization

The application ensures that the same preprocessing pipeline used during training is applied during inference.

---

## 6. Repository Structure

project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
│-- logistic_regression.pkl
│-- decision_tree.pkl
│-- knn.pkl
│-- naive_bayes.pkl
│-- random_forest.pkl
│-- xgboost.pkl
│-- scaler.pkl
│-- label_encoder.pkl

---

## 7. Deployment
The application was deployed on **Streamlit Community Cloud** and can be accessed via the provided live link.
