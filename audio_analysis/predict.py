import joblib
import numpy as np
import os
from feature_extraction import extract_features


# -----------------------------
# LOAD MODEL + SCALER
# -----------------------------
BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "emotion_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)


# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict_emotion(file_path):

    print(f"[INFO] Processando áudio: {file_path}")

    features = extract_features(file_path)

    if features is None:
        return {
            "error": "Não foi possível extrair features"
        }

    # garantir formato correto
    features = np.array(features)

    # -----------------------------
    # SCALER FIX (IMPORTANTE)
    # -----------------------------
    features = scaler.transform([features])

    # -----------------------------
    # PREDICTION
    # -----------------------------
    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]
    confidence = float(np.max(probabilities))

    return {
        "emotion": str(prediction),
        "confidence": confidence
    }


# -----------------------------
# TEST
# -----------------------------
if __name__ == "__main__":

    test_file = os.path.join(
        BASE_DIR,
        "dataset",
        "Actor_01",
        "03-01-06-01-02-01-01.wav"
    )

    result = predict_emotion(test_file)

    print(result)