from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)

# ===================================================
# 1. Charger le modèle et le scaler
# ===================================================
MODEL_PATH = "models/model_info_complete.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Le fichier {MODEL_PATH} n'existe pas. Assurez-vous d'avoir exécuté le script d'entraînement.")

with open(MODEL_PATH, "rb") as f:
    model_info = pickle.load(f)

model = model_info["model"]
scaler = model_info["scaler"]
test_mse = model_info["test_mse"]
test_r2 = model_info["test_r2"]

print("=" * 50)
print("MODELE CHARGE AVEC SUCCES")
print("=" * 50)
print(f"Test MSE: {test_mse:.4f}")
print(f"Test R2: {test_r2:.4f}")
print(f"Learning Rate: {model_info['learning_rate']}")
print(f"Batch Size: {model_info['batch_size']}")
print("=" * 50)

# ===================================================
# 2. Route principale (Page HTML)
# ===================================================
@app.route('/')
def home():
    return render_template('index.html')

# ===================================================
# 3. Route d'information sur le modèle
# ===================================================
@app.route('/api/model-info', methods=['GET'])
def model_info_endpoint():
    """Retourne les informations sur le modèle"""
    return jsonify({
        "test_mse": float(test_mse),
        "test_r2": float(test_r2),
        "learning_rate": float(model_info["learning_rate"]),
        "batch_size": int(model_info["batch_size"]) if model_info["batch_size"] else "Batch GD",
        "n_epochs": int(model.n_epochs)
    })

# ===================================================
# 4. Route de prédiction
# ===================================================
@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Effectue une prédiction de prix dynamique
    
    Input JSON:
    {
        "demand_index": float,
        "competition_pressure": float,
        "operational_cost": float,
        "seasonality_index": float,
        "marketing_intensity": float,
        "time_slot": int (0-23),
        "day_of_week": int (0-6, 0=Lundi)
    }
    """
    try:
        data = request.get_json()
        
        # Validation des données - tous les champs sont maintenant optionnels
        # On utilise des valeurs par défaut si non fournies
        demand_index = float(data.get("demand_index", 0))
        competition_pressure = float(data.get("competition_pressure", 0))
        operational_cost = float(data.get("operational_cost", 0))
        seasonality_index = float(data.get("seasonality_index", 0))
        marketing_intensity = float(data.get("marketing_intensity", 0))
        time_slot = int(data.get("time_slot", 0))
        day_of_week = int(data.get("day_of_week", 0))
        
        # Validation des ranges
        if not (0 <= time_slot <= 23):
            return jsonify({"error": "time_slot doit être entre 0 et 23"}), 400
        if not (0 <= day_of_week <= 6):
            return jsonify({"error": "day_of_week doit être entre 0 et 6"}), 400
        
        # Encodage cyclique
        time_slot_sin = np.sin(2 * np.pi * time_slot / 24)
        time_slot_cos = np.cos(2 * np.pi * time_slot / 24)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)
        
        # Construction du vecteur de features
        continuous_features = np.array([
            demand_index, competition_pressure, operational_cost,
            seasonality_index, marketing_intensity
        ]).reshape(1, -1)
        
        cyclical_features = np.array([
            time_slot_sin, time_slot_cos, day_sin, day_cos
        ]).reshape(1, -1)
        
        # Scaling des features continues uniquement
        continuous_scaled = scaler.transform(continuous_features)
        
        # Combinaison des features
        X = np.concatenate([continuous_scaled, cyclical_features], axis=1)
        
        # Prédiction
        prediction = model.predict(X)
        predicted_price = float(prediction[0, 0])
        
        # Réponse
        return jsonify({
            "predicted_price": round(predicted_price, 2),
            "input_features": {
                "demand_index": demand_index,
                "competition_pressure": competition_pressure,
                "operational_cost": operational_cost,
                "seasonality_index": seasonality_index,
                "marketing_intensity": marketing_intensity,
                "time_slot": time_slot,
                "day_of_week": day_of_week
            }
        })
        
    except ValueError as e:
        return jsonify({"error": f"Erreur de validation: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Erreur serveur: {str(e)}"}), 500

# ===================================================
# 5. Route de prédiction par batch
# ===================================================
@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    """
    Effectue des prédictions pour plusieurs entrées
    
    Input JSON:
    {
        "predictions": [
            {...}, {...}, ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if "predictions" not in data or not isinstance(data["predictions"], list):
            return jsonify({"error": "Format invalide. Attendu: {'predictions': [...]}"}), 400
        
        results = []
        
        for idx, item in enumerate(data["predictions"]):
            try:
                # Réutiliser la logique de prédiction unique
                request.json = item
                response = predict()
                
                if response[1] == 200:  # Success
                    results.append({
                        "index": idx,
                        "success": True,
                        "result": response[0].get_json()
                    })
                else:
                    results.append({
                        "index": idx,
                        "success": False,
                        "error": response[0].get_json()
                    })
            except Exception as e:
                results.append({
                    "index": idx,
                    "success": False,
                    "error": str(e)
                })
        
        return jsonify({
            "total": len(results),
            "successful": sum(1 for r in results if r["success"]),
            "results": results
        })
        
    except Exception as e:
        return jsonify({"error": f"Erreur serveur: {str(e)}"}), 500

# ===================================================
# 6. Health check
# ===================================================
@app.route('/api/health', methods=['GET'])
def health():
    """Vérification de l'état du service"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    })

# ===================================================
# 7. Lancement de l'application
# ===================================================
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)