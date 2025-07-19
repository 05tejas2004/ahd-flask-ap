
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("adhd_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = [
        data["age"],
        data["gender"],
        data["inattention"],
        data["hyperactivity"],
        data["impulsivity"],
        data["cognitive_score"]
    ]
    X = scaler.transform([features])
    prob = model.predict_proba(X)[0][1]
    return jsonify({"adhd_risk_percent": round(prob * 100, 2)})

if __name__ == "__main__":
    app.run(debug=True)
