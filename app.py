import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model
model = joblib.load("fraud_xgboost_model.pkl")

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection System")
st.write("Predict whether a transaction is Fraudulent or Safe.")

# =========================
# Batch Prediction Section
# =========================

st.markdown("## 📂 Batch Prediction (CSV Upload)")

uploaded_file = st.file_uploader(
    "Upload CSV file containing transaction features",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        st.write("### Uploaded Data Preview")
        st.dataframe(data.head())

        probs = model.predict_proba(data)[:, 1]
        preds = (probs > 0.3).astype(int)

        result_df = data.copy()
        result_df["Fraud Probability"] = probs
        result_df["Prediction"] = np.where(preds == 1, "Fraud", "Safe")

        st.write("### Prediction Results")
        st.dataframe(result_df.head())

        fraud_count = int(sum(preds))
        safe_count = int(len(preds) - fraud_count)

        st.write("### Summary")
        st.bar_chart({
            "Fraud": [fraud_count],
            "Safe": [safe_count]
        })

    except Exception as e:
        st.error(f"Prediction Error: {e}")

st.divider()

# =========================
# Single Prediction Section
# =========================

st.markdown("## 🔍 Single Transaction Prediction")

feature_input = st.text_area(
    f"Enter {model.n_features_in_} features separated by commas",
    height=150
)

if st.button("Predict Transaction"):

    try:
        features = np.array(
            [float(x.strip()) for x in feature_input.split(",")]
        ).reshape(1, -1)

        if features.shape[1] != model.n_features_in_:
            st.error(
                f"Expected {model.n_features_in_} features but received {features.shape[1]}"
            )

        else:
            prob = model.predict_proba(features)[0][1]
            prediction = 1 if prob > 0.3 else 0

            st.subheader("Prediction Result")

            st.write(
                f"**Fraud Probability:** {prob:.4f}"
            )

            if prediction == 1:
                st.error("🚨 Fraudulent Transaction Detected")
            else:
                st.success("✅ Safe Transaction")

            st.markdown("### Probability Visualization")

            st.progress(min(int(prob * 100), 100))

            chart_df = pd.DataFrame({
                "Probability": [prob, 1 - prob]
            }, index=["Fraud", "Safe"])

            st.bar_chart(chart_df)

    except ValueError:
        st.error(
            "Please enter only numeric values separated by commas."
        )

    except Exception as e:
        st.error(f"Error: {e}")

