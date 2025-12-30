


import numpy as np
import pandas as pd
import pickle

# -----------------------------
# 1. Charger modèle et scaler
# -----------------------------
with open("results/best_model_lr0.01_bs1.pkl", "rb") as f:
    best_model = pickle.load(f)

with open("results/scaler_lr0.01_bs1.pkl", "rb") as f:
    scaler = pickle.load(f)

# -----------------------------
# 2. Nouvelles données
# -----------------------------
data = pd.DataFrame({
    "demand_index": [89.0, 56.0],
    "time_slot": [23.0, 4.0],
    "day_of_week": [3.0, 6.0],
    "competition_pressure": [0.09, 0.98],
    "operational_cost": [99.70, 184.11],
    "seasonality_index": [0.88, 0.55],
    "marketing_intensity": [0.547835, 0.880000],
    "dynamic_price": [363.653116, 440.052599]  # vrai prix pour comparaison
})

# -----------------------------
# 3. Encodage cyclique
# -----------------------------
data["time_slot_sin"] = np.sin(2 * np.pi * data["time_slot"] / 24)
data["time_slot_cos"] = np.cos(2 * np.pi * data["time_slot"] / 24)
data["day_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
data["day_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)

data = data.drop(columns=["time_slot", "day_of_week"])

# -----------------------------
# 4. Features & Target
# -----------------------------
continuous_features = [
    "demand_index",
    "competition_pressure",
    "operational_cost",
    "seasonality_index",
    "marketing_intensity"
]

cyclical_features = [
    "time_slot_sin",
    "time_slot_cos",
    "day_sin",
    "day_cos"
]

X_new = data[continuous_features + cyclical_features].values
y_true = data["dynamic_price"].values.reshape(-1, 1)

# -----------------------------
# 5. Scaling (comme en training)
# -----------------------------
X_new[:, :len(continuous_features)] = scaler.transform(X_new[:, :len(continuous_features)])

# -----------------------------
# 6. Prédiction
# -----------------------------
y_pred = best_model.predict(X_new)

# -----------------------------
# 7. Résultats
# -----------------------------
for i in range(len(y_true)):
    print(f"Exemple {i}: vrai={y_true[i][0]:.2f}, prédit={y_pred[i][0]:.2f}")