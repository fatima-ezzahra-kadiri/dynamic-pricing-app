
 
 
# import numpy as np
# import pandas as pd
# import pickle
# import os
# import matplotlib.pyplot as plt

# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from gradient_descent import GradientDescentRegressor


# # ===================================================
# # 1. Load data
# # ===================================================
# df = pd.read_csv("df_clean_without_target_outliers.csv")


# # ===================================================
# # 2. Cyclical encoding
# # ===================================================
# df["time_slot_sin"] = np.sin(2 * np.pi * df["time_slot"] / 24)
# df["time_slot_cos"] = np.cos(2 * np.pi * df["time_slot"] / 24)

# df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
# df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

# df = df.drop(columns=["time_slot", "day_of_week"])


# # ===================================================
# # 3. Features & Target
# # ===================================================
# target = "dynamic_price"

# continuous_features = [
#     "demand_index",
#     "competition_pressure",
#     "operational_cost",
#     "seasonality_index",
#     "marketing_intensity"
# ]

# cyclical_features = [
#     "time_slot_sin",
#     "time_slot_cos",
#     "day_sin",
#     "day_cos"
# ]

# X = df[continuous_features + cyclical_features].values
# y = df[target].values.reshape(-1, 1)


# # ===================================================
# # 4. Train / Validation / Test split
# # ===================================================
# X_temp, X_test, y_temp, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# X_train, X_val, y_train, y_val = train_test_split(
#     X_temp, y_temp, test_size=0.25, random_state=42
# )  # 60% train / 20% val / 20% test


# # ===================================================
# # 5. Scaling (continuous only)
# # ===================================================
# scaler = StandardScaler()

# X_train[:, :len(continuous_features)] = scaler.fit_transform(
#     X_train[:, :len(continuous_features)]
# )
# X_val[:, :len(continuous_features)] = scaler.transform(
#     X_val[:, :len(continuous_features)]
# )
# X_test[:, :len(continuous_features)] = scaler.transform(
#     X_test[:, :len(continuous_features)]
# )


# # ===================================================
# # 6. Metrics
# # ===================================================
# def mse(y_true, y_pred):
#     return np.mean((y_true - y_pred) ** 2)


# def r2_score(y_true, y_pred):
#     ss_res = np.sum((y_true - y_pred) ** 2)
#     ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
#     return 1 - ss_res / ss_tot


# # ===================================================
# # 7. Hyperparameters
# # ===================================================
# learning_rates = [0.0001, 0.001, 0.005, 0.01]
# batch_sizes = [1, 16, 32, 64, 70, None]

# results = []
# best_val_mse = float("inf")
# best_model = None
# best_params = {}

# os.makedirs("results", exist_ok=True)

# print("\nHyperparameter tuning with Validation set...\n")


# # ===================================================
# # 8. Validation loop
# # ===================================================
# for lr in learning_rates:
#     for bs in batch_sizes:

#         effective_bs = X_train.shape[0] if bs is None else bs

#         model = GradientDescentRegressor(
#             learning_rate=lr,
#             n_epochs=3000,
#             batch_size=effective_bs
#         )

#         model.fit(X_train, y_train)

#         y_val_pred = model.predict(X_val)

#         val_mse = mse(y_val, y_val_pred)
#         val_r2 = r2_score(y_val, y_val_pred)

#         method = (
#             "Batch GD" if bs is None else
#             "SGD" if bs == 1 else
#             f"Mini-Batch ({bs})"
#         )

#         print(
#             f"LR={lr:<7} | {method:<18} | "
#             f"Val MSE={val_mse:.4f} | Val R2={val_r2:.4f}"
#         )

#         results.append({
#             "learning_rate": lr,
#             "batch_size": effective_bs,
#             "val_mse": val_mse,
#             "val_r2": val_r2
#         })

#         if val_mse < best_val_mse:
#             best_val_mse = val_mse
#             best_model = model
#             best_params = {
#                 "learning_rate": lr,
#                 "batch_size": bs,
#                 "method": method,
#                 "val_r2": val_r2
#             }


# # ===================================================
# # 9. Final evaluation on TEST set
# # ===================================================
# y_test_pred = best_model.predict(X_test)

# test_mse = mse(y_test, y_test_pred)
# test_r2 = r2_score(y_test, y_test_pred)

# print("\nFinal Test Performance")
# print(f"Test MSE: {test_mse:.4f}")
# print(f"Test R2 : {test_r2:.4f}")


# # ===================================================
# # 10. Save model & scaler
# # ===================================================
# with open("results/best_model_without_target_outliers2.pkl", "wb") as f:
#     pickle.dump(best_model, f)

# with open("results/scaler_without_target_outliers2.pkl", "wb") as f:
#     pickle.dump(scaler, f)


# # ===================================================
# # 11. Visualisations 
# # ===================================================
# results_df = pd.DataFrame(results)


# # --- (1) Validation MSE vs Batch Size
# plt.figure()
# for lr in results_df["learning_rate"].unique():
#     subset = results_df[results_df["learning_rate"] == lr]
#     subset = subset.sort_values("batch_size")
#     plt.plot(subset["batch_size"], subset["val_mse"], marker="o", label=f"LR={lr}")

# plt.xlabel("Batch Size")
# plt.ylabel("Validation MSE")
# plt.title("Validation MSE vs Batch Size")
# plt.ylim(0, 15000)  #  ZOOM MSE
# plt.legend()
# plt.grid(True)
# plt.show()


# # --- (2) Predicted vs True (TEST)
# plt.figure()
# plt.scatter(y_test, y_test_pred, alpha=0.6)
# plt.plot(
#     [y_test.min(), y_test.max()],
#     [y_test.min(), y_test.max()],
#     linestyle="--"
# )
# plt.xlabel("True Price")
# plt.ylabel("Predicted Price")
# plt.title("Predicted vs True Prices (Test Set)")
# plt.grid(True)
# plt.show()


# # --- (3) Residual distribution
# residuals = y_test - y_test_pred

# plt.figure()
# plt.hist(residuals, bins=40)
# plt.xlabel("Residuals")
# plt.ylabel("Frequency")
# plt.title("Residual Distribution (Test Set)")
# plt.grid(True)
# plt.show()


# # --- (4) Heatmap LR x Batch Size (Val MSE)
# pivot = results_df.pivot(
#     index="batch_size",
#     columns="learning_rate",
#     values="val_mse"
# )

# plt.figure()
# plt.imshow(pivot, aspect="auto", vmin=0, vmax=15000)  #  ZOOM MSE
# plt.colorbar(label="Validation MSE (0–15000)")
# plt.xticks(range(len(pivot.columns)), pivot.columns)
# plt.yticks(range(len(pivot.index)), pivot.index)
# plt.xlabel("Learning Rate")
# plt.ylabel("Batch Size")
# plt.title("Hyperparameter Sensitivity (Validation MSE)")
# plt.show()


# # ===================================================
# # 12. Summary
# # ===================================================
# print("\nBest Model Found (Validation)")
# for k, v in best_params.items():
#     print(f"{k}: {v}")
# print(f"Best Val MSE: {best_val_mse:.4f}")





import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from gradient_descent import GradientDescentRegressor

# ===================================================
# 1. Load data
# ===================================================
df = pd.read_csv("df_clean_without_target_outliers.csv")

# ===================================================
# 2. Cyclical encoding
# ===================================================
df["time_slot_sin"] = np.sin(2 * np.pi * df["time_slot"] / 24)
df["time_slot_cos"] = np.cos(2 * np.pi * df["time_slot"] / 24)
df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
df = df.drop(columns=["time_slot", "day_of_week"])

# ===================================================
# 3. Features & Target
# ===================================================
target = "dynamic_price"
continuous_features = [
    "demand_index", "competition_pressure", "operational_cost",
    "seasonality_index", "marketing_intensity"
]
cyclical_features = [
    "time_slot_sin", "time_slot_cos", "day_sin", "day_cos"
]

X = df[continuous_features + cyclical_features].values
y = df[target].values.reshape(-1, 1)

# ===================================================
# 4. Train / Validation / Test split
# ===================================================
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# ===================================================
# 5. Scaling (continuous only)
# ===================================================
scaler = StandardScaler()
X_train[:, :len(continuous_features)] = scaler.fit_transform(X_train[:, :len(continuous_features)])
X_val[:, :len(continuous_features)] = scaler.transform(X_val[:, :len(continuous_features)])
X_test[:, :len(continuous_features)] = scaler.transform(X_test[:, :len(continuous_features)])

# ===================================================
# 6. Metrics
# ===================================================
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / ss_tot

# ===================================================
# 7. Hyperparameters
# ===================================================
learning_rates = [0.0001, 0.001, 0.005, 0.01]
batch_sizes = [1, 16, 32, 64, 70, None]  # None = Batch GD

results = []
best_val_mse = float("inf")
best_model = None
best_params = {}
os.makedirs("results", exist_ok=True)

print("\nHyperparameter tuning with Validation set...\n")

# ===================================================
# 8. Validation loop
# ===================================================
for lr in learning_rates:
    for bs in batch_sizes:
        effective_bs = X_train.shape[0] if bs is None else bs
        model = GradientDescentRegressor(
            learning_rate=lr,
            n_epochs=3000,
            batch_size=effective_bs
        )
        # Enregistrement learning curves par epoch
        train_curve = []
        val_curve = []
        for epoch in range(model.n_epochs):
            model.partial_fit(X_train, y_train)
            train_curve.append(mse(y_train, model.predict(X_train)))
            val_curve.append(mse(y_val, model.predict(X_val)))
        
        val_mse = val_curve[-1]
        val_r2 = r2_score(y_val, model.predict(X_val))
        method = "Batch GD" if bs is None else "SGD" if bs==1 else f"Mini-Batch ({bs})"
        print(f"LR={lr:<7} | {method:<18} | Val MSE={val_mse:.4f} | Val R2={val_r2:.4f}")
        
        results.append({
            "learning_rate": lr,
            "batch_size": effective_bs,
            "val_mse": val_mse,
            "val_r2": val_r2,
            "train_curve": train_curve,
            "val_curve": val_curve
        })
        
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_model = model
            best_params = {
                "learning_rate": lr,
                "batch_size": bs,
                "method": method,
                "val_r2": val_r2,
                "train_curve": train_curve,
                "val_curve": val_curve
            }

results_df = pd.DataFrame(results)

# ===================================================
# 9. Final evaluation on TEST set
# ===================================================
y_test_pred = best_model.predict(X_test)
test_mse = mse(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
print("\nFinal Test Performance")
print(f"Test MSE: {test_mse:.4f}, Test R2: {test_r2:.4f}")

# ===================================================
# 10. Save model, scaler & metrics
# ===================================================
model_info = {
    "model": best_model,
    "scaler": scaler,
    "test_mse": test_mse,
    "test_r2": test_r2,
    "learning_rate": best_params["learning_rate"],
    "batch_size": best_params["batch_size"],
    "train_curve": best_params["train_curve"],
    "val_curve": best_params["val_curve"]
}

with open("results/model_info_complete.pkl", "wb") as f:
    pickle.dump(model_info, f)

# ===================================================
# 11. Visualisations
# ===================================================

# --- (1) Learning curves Train vs Validation
plt.figure(figsize=(8,5))
plt.plot(best_params["train_curve"], label="Train MSE")
plt.plot(best_params["val_curve"], label="Validation MSE")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("Learning Curves (Train vs Validation)")
plt.legend()
plt.grid(True)
plt.show()

# --- (2) Predicted vs True Prices (Test Set)
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_test_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle="--", color="red")
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("Predicted vs True Prices (Test Set)")
plt.grid(True)
plt.show()

# --- (3) Residual Distribution
residuals = y_test - y_test_pred
plt.figure()
plt.hist(residuals, bins=40)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Residual Distribution (Test Set)")
plt.grid(True)
plt.show()

# --- (4) Validation MSE vs Batch Size (pour chaque LR)
plt.figure()
for lr in results_df["learning_rate"].unique():
    subset = results_df[results_df["learning_rate"] == lr].sort_values("batch_size")
    plt.plot(subset["batch_size"], subset["val_mse"], marker="o", label=f"LR={lr}")
plt.xlabel("Batch Size")
plt.ylabel("Validation MSE")
plt.ylim(0, 15000)
plt.title("Validation MSE vs Batch Size")
plt.legend()
plt.grid(True)
plt.show()

# --- (5) Heatmap Validation MSE
pivot = results_df.pivot(index="batch_size", columns="learning_rate", values="val_mse")
plt.figure()
plt.imshow(pivot, aspect="auto", vmin=0, vmax=15000)
plt.colorbar(label="Validation MSE")
plt.xticks(range(len(pivot.columns)), pivot.columns)
plt.yticks(range(len(pivot.index)), pivot.index)
plt.xlabel("Learning Rate")
plt.ylabel("Batch Size")
plt.title("Hyperparameter Sensitivity (Validation MSE)")
plt.show()

 