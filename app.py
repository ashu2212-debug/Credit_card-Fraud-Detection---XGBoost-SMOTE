print("HELLO FROM APP")

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("fraud_xgboost_model.pkl")

@app.route("/")
def home():
    return "Credit card fraud detection API running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = np.array(data["features"]).reshape(1, -1)

    probability = model.predict_proba(features)[0][1]

    prediction = 1 if probability > 0.3 else 0

    return jsonify({
        "fraud_probability": float(probability),
        "prediction": int(prediction)
    })

if __name__ == "__main__":
    app.run(debug=True)