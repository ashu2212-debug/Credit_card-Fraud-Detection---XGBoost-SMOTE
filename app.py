import streamlit as st
import joblib
import numpy as np

model = joblib.load("fraud_xgboost_model.pkl")

st.title("Credit Card Fraud Detection")

feature_input = st.text_input(
    "Enter features separated by commas"
)

if st.button("Predict"):
    features = np.array(
        [float(x) for x in feature_input.split(",")]
    ).reshape(1, -1)

    prob = model.predict_proba(features)[0][1]

    prediction = 1 if prob > 0.3 else 0

    st.write("Fraud Probability:", prob)
    st.write("Prediction:", prediction)
