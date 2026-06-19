import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model
model = joblib.load("fraud_xgboost_model.pkl")

st.title("Credit Card Fraud Detection")


    # Visualization
st.progress(int(prob * 100))
st.bar_chart({"Fraud Probability": [prob], "Safe Probability": [1 - prob]})

# Batch prediction via CSV upload
st.markdown("### 🔹 Batch Prediction (Upload CSV)")
uploaded_file = st.file_uploader("Upload CSV file with features", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", data.head())

    try:
        probs = model.predict_proba(data)[:, 1]
        preds = (probs > 0.3).astype(int)

        result_df = data.copy()
        result_df["Fraud Probability"] = probs
        result_df["Prediction"] = preds

        st.write("Batch Prediction Results:", result_df.head())
        st.bar_chart({"Fraud": [sum(preds)], "Safe": [len(preds) - sum(preds)]})

    except Exception as e:
        st.error(f"Error in prediction: {e}")


st.markdown("### 🔹 Single Transaction Prediction")

feature_input = st.text_area("Enter 30 features separated by commas")

if st.button("Predict"):
    try:
        features = np.array([float(x.strip()) for x in feature_input.split(",")]).reshape(1, -1)
        if features.shape[1] != model.n_features_in_:
            st.error(f"Expected {model.n_features_in_} features, but got {features.shape[1]}.")
        else:
            prob = model.predict_proba(features)[0][1]
            prediction = 1 if prob > 0.3 else 0
            st.write("Fraud Probability:", prob)
            st.write("Prediction:", "Fraud" if prediction == 1 else "Safe")
    except ValueError:
        st.error("Please enter only numeric values separated by commas.")

