import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Synthetic training data with meaningful fraud signals:
# Features: amount, amount_deviation (ratio to user avg), is_new_location,
#           is_flagged_device (shared by 3+ users), transaction_velocity (last 2 min)
data = {
    "amount":            [100,  200,  50,   300,  150,  75,   250,  400,  180,  90,
                          5000, 7000, 10000, 8000, 6000, 9000, 5500, 7500, 6500, 8500],
    "amount_deviation":  [1.0,  1.1,  0.9,  1.2,  1.0,  0.8,  1.3,  1.5,  1.1,  0.9,
                          10.0, 15.0, 20.0, 12.0, 8.0,  18.0, 11.0, 14.0, 9.0,  16.0],
    "is_new_location":   [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                          1,    1,    1,    1,    0,    1,    1,    0,    1,    1],
    "is_flagged_device": [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                          0,    1,    1,    0,    1,    1,    0,    1,    0,    1],
    "velocity":          [1,    2,    1,    3,    2,    1,    2,    1,    3,    2,
                          1,    2,    8,    6,    10,   7,    3,    5,    4,    9],
    "fraud":             [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                          1,    1,    1,    1,    1,    1,    1,    1,    1,    1],
}

df = pd.DataFrame(data)

X = df[["amount", "amount_deviation", "is_new_location", "is_flagged_device", "velocity"]]
y = df["fraud"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "app/ML/model.pkl")
print("Model trained and saved to app/ML/model.pkl")
