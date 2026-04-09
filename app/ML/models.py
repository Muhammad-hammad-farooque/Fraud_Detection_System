import joblib
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(MODEL_PATH)

def predict_fraud(amount: float, amount_deviation: float, is_new_location: int,
                  is_flagged_device: int, velocity: int):
    """
    Returns (prediction: int, fraud_probability: float).
    Features must match train_model.py exactly.
    """
    features = np.array([[amount, amount_deviation, is_new_location, is_flagged_device, velocity]])
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])
    return prediction, probability
