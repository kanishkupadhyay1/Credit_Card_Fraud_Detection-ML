import joblib
import numpy as np

# Load trained model
model = joblib.load("../models/fraud_detection_model.pkl")

# Load scaler
scaler = joblib.load("../models/scaler.pkl")

# Example transaction with 30 features
transaction = np.random.rand(1, 30)

# Scale transaction
transaction_scaled = scaler.transform(transaction)

# Predict
prediction = model.predict(transaction_scaled)

if prediction[0] == 1:
    print("⚠️ Fraud Transaction Detected")
else:
    print("✅ Normal Transaction")