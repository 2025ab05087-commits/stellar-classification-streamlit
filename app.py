import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score
)

FEATURE_COLUMNS = [
    'alpha', 'delta', 'u', 'g', 'r', 'i', 'z',
    'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'redshift'
]


# App Title
st.title("Stellar Classification using Machine Learning")
st.write(
    "This application classifies astronomical objects into **Star, Galaxy, or Quasar** "
    "using multiple machine learning models."
)

# Load Models
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "kNN": joblib.load("model/knn.pkl"),
        "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
        "Random Forest": joblib.load("model/random_forest.pkl"),
        "XGBoost": joblib.load("model/xgboost.pkl")
    }
    scaler = joblib.load("model/scaler.pkl")
    label_encoder = joblib.load("model/label_encoder.pkl")
    return models, scaler, label_encoder

models, scaler, label_encoder = load_models()

# Upload CSV
st.header("Upload Test Dataset")
uploaded_file = st.file_uploader("Upload CSV file (test data only)", type=["csv"])
if uploaded_file is not None:
    st.download_button(
        label="Download Uploaded Test CSV",
        data=uploaded_file.getvalue(),
        file_name=uploaded_file.name,
        mime="text/csv"
    )

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(data.head())

    # Model Selection
    model_name = st.selectbox("Select Model", list(models.keys()))
    model = models[model_name]

    # Prediction
    if st.button("Run Prediction"):
        try:
            data_copy = data.copy()

            y_true = None
            if 'class' in data_copy.columns:
                y_true = label_encoder.transform(data_copy['class'])
                data_copy = data_copy.drop(columns=['class'])

            X = data_copy[FEATURE_COLUMNS]

            X_scaled = scaler.transform(X)

            y_pred = model.predict(X_scaled)
            y_prob = model.predict_proba(X_scaled)

            y_pred_labels = label_encoder.inverse_transform(y_pred)

            st.subheader("Prediction Results")
            st.write(pd.Series(y_pred_labels).value_counts())

            if y_true is not None:
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, average="macro")
                rec = recall_score(y_true, y_pred, average="macro")
                f1 = f1_score(y_true, y_pred, average="macro")
                mcc = matthews_corrcoef(y_true, y_pred)
                auc = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="macro"
                )

                st.subheader("Evaluation Metrics")
                st.write({
                    "Accuracy": acc,
                    "AUC": auc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1 Score": f1,
                    "MCC": mcc
                })

                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)

                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

            else:
                st.info(
                    "True labels not found in uploaded file. "
                    "Metrics and confusion matrix are not shown."
                )

        except Exception as e:
            st.error(f"Error during prediction: {e}")

