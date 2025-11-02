import pickle
import numpy as np
import pandas as pd

# =============================
# 1. Load saved components
# =============================

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# =============================
# 2. Define prediction function
# =============================

def predict_intrusion(new_data):
    """
    Takes a single flow (list, NumPy array, or DataFrame row)
    and predicts the type of attack or 'BENIGN'.
    """
    # Convert to NumPy array & reshape
    new_data = np.array(new_data).reshape(1, -1)

    # Scale input
    new_data_scaled = scaler.transform(new_data)

    # Predict class
    pred_class = model.predict(new_data_scaled)[0]
    pred_prob = model.predict_proba(new_data_scaled)[0]

    # Decode label
    attack_label = label_encoder.inverse_transform([pred_class])[0]
    confidence = round(np.max(pred_prob) * 100, 2)

    print(f"Predicted Attack Type: {attack_label}")
    print(f"Confidence: {confidence}%")
    return attack_label, confidence


# =============================
# 3. Example usage
# =============================

if __name__ == "__main__":
    # Load a sample flow from your test CSV
    df = pd.read_csv("X_test.csv")

    # Choose one record to predict
    sample = df.iloc[0].values  # first record

    # Run prediction
    predict_intrusion(sample)
