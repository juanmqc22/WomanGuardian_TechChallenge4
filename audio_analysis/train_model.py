import os
import numpy as np
import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from feature_extraction import extract_features


DATASET_PATH = "dataset"

emotion_map = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

X = []
y = []
total_files = 0
ignored_files = 0

# -----------------------------
# DATASET LOADING
# -----------------------------
for actor_folder in sorted(os.listdir(DATASET_PATH)):
    actor_path = os.path.join(DATASET_PATH, actor_folder)

    if not os.path.isdir(actor_path):
        continue

    print(f"[INFO] Processando {actor_folder}")

    for file_name in os.listdir(actor_path):
        if not file_name.endswith(".wav"):
            continue

        total_files += 1
        parts = file_name.split("-")

        if len(parts) < 3:
            ignored_files += 1
            continue

        emotion_code = parts[2]

        if emotion_code not in emotion_map:
            ignored_files += 1
            continue

        emotion = emotion_map[emotion_code]
        file_path = os.path.join(actor_path, file_name)

        features = extract_features(file_path)

        if features is None:
            ignored_files += 1
            continue

        X.append(features)
        y.append(emotion)

# -----------------------------
# VALIDAÇÃO DE TAMANHO
# -----------------------------
sizes = [len(f) for f in X]
expected_size = max(set(sizes), key=sizes.count)  # tamanho mais comum

X_clean, y_clean = [], []
for features, label in zip(X, y):
    if len(features) == expected_size:
        X_clean.append(features)
        y_clean.append(label)
    else:
        print(f"[SKIP] Feature com tamanho inesperado: {len(features)} (esperado {expected_size})")
        ignored_files += 1

print(f"\nTotal válidos: {len(X_clean)} | Ignorados: {ignored_files}")
print(f"Tamanho do vetor de features: {expected_size}\n")

# -----------------------------
# CONVERSÃO SEGURA
# -----------------------------
X_arr = np.array(X_clean, dtype=np.float64)
y_arr = np.array(y_clean)

# Sanidade
assert X_arr.dtype == np.float64, "dtype incorreto!"
assert not np.any(np.isnan(X_arr)), "Ainda há NaN!"
assert not np.any(np.isinf(X_arr)), "Ainda há Inf!"

print(f"[OK] Shape: {X_arr.shape}, dtype: {X_arr.dtype}")

# -----------------------------
# NORMALIZAÇÃO
# -----------------------------
scaler = StandardScaler()
X_arr = scaler.fit_transform(X_arr).astype(np.float64)

# -----------------------------
# SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_arr, y_arr, test_size=0.2, random_state=42, stratify=y_arr
)

# -----------------------------
# MODELO
# -----------------------------
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    verbose=1
)

print("[INFO] Treinando modelo...")
model.fit(X_train, y_train)

# -----------------------------
# AVALIAÇÃO
# -----------------------------
accuracy = model.score(X_test, y_test)
print(f"\nAcurácia: {accuracy:.2f}\n")
print(classification_report(y_test, model.predict(X_test)))

# -----------------------------
# SALVAR
# -----------------------------
joblib.dump(model, "emotion_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("[OK] Modelo e scaler salvos!")