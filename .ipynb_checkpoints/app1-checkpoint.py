from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)  # Permet les requêtes depuis le frontend

# ===================================================
# Chargement du modèle au démarrage
# ===================================================
MODEL_PATH = "results/model_info.pkl"
model_info = None

def load_model():
    """Charge le modèle et ses métadonnées"""
    global model_info
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Le fichier {MODEL_PATH} n'existe pas")
        
        with open(MODEL_PATH, "rb") as f:
            model_info = pickle.load(f)
        
        print("✓ Modèle chargé avec succès!")
        print(f"  - Test MSE: {model_info['test_mse']:.2f}")
        print(f"  - Test R²: {model_info['test_r2']:.4f}")
        
    except Exception as e:
        print(f"✗ Erreur lors du chargement du modèle: {e}")
        raise

# Charger le modèle au démarrage
load_model()

# ===================================================
# Routes API
# ===================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Vérifie si l'API fonctionne"""
    return jsonify({
        "status": "ok",
        "message": "API is running",
        "model_loaded": model_info is not None
    }), 200


@app.route('/model/info', methods=['GET'])
def get_model_info():
    """Retourne les informations du modèle (MSE, R²)"""
    if model_info is None:
        return jsonify({"error": "Modèle non chargé"}), 500
    
    return jsonify({
        "test_mse": float(model_info['test_mse']),
        "test_r2": float(model_info['test_r2']),
        "metrics": {
            "mse": float(model_info['test_mse']),
            "r2": float(model_info['test_r2'])
        }
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prédit le prix dynamique à partir des features
    
    Body JSON attendu:
    {
        "demand_index": float,
        "competition_pressure": float,
        "operational_cost": float,
        "seasonality_index": float,
        "marketing_intensity": float,
        "time_slot": int (0-23),
        "day_of_week": int (0-6)
    }
    """
    if model_info is None:
        return jsonify({"error": "Modèle non chargé"}), 500
    
    try:
        data = request.get_json()
        
        # Validation des données
        required_fields = [
            "demand_index", "competition_pressure", "operational_cost",
            "seasonality_index", "marketing_intensity", "time_slot", "day_of_week"
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Champ manquant: {field}"}), 400
        
        # Extraction des features
        demand_index = float(data["demand_index"])
        competition_pressure = float(data["competition_pressure"])
        operational_cost = float(data["operational_cost"])
        seasonality_index = float(data["seasonality_index"])
        marketing_intensity = float(data["marketing_intensity"])
        time_slot = int(data["time_slot"])
        day_of_week = int(data["day_of_week"])
        
        # Validation des valeurs
        if not (0 <= time_slot <= 23):
            return jsonify({"error": "time_slot doit être entre 0 et 23"}), 400
        if not (0 <= day_of_week <= 6):
            return jsonify({"error": "day_of_week doit être entre 0 et 6"}), 400
        
        # Encodage cyclique
        time_slot_sin = np.sin(2 * np.pi * time_slot / 24)
        time_slot_cos = np.cos(2 * np.pi * time_slot / 24)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)
        
        # Construction du vecteur de features (ordre important!)
        continuous_features = np.array([
            demand_index,
            competition_pressure,
            operational_cost,
            seasonality_index,
            marketing_intensity
        ]).reshape(1, -1)
        
        cyclical_features = np.array([
            time_slot_sin,
            time_slot_cos,
            day_sin,
            day_cos
        ]).reshape(1, -1)
        
        # Scaling des features continues uniquement
        scaler = model_info['scaler']
        continuous_features_scaled = scaler.transform(continuous_features)
        
        # Combinaison des features
        X = np.concatenate([continuous_features_scaled, cyclical_features], axis=1)
        
        # Prédiction
        model = model_info['model']
        predicted_price = model.predict(X)[0, 0]
        
        # Réponse
        return jsonify({
            "predicted_price": float(predicted_price),
            "model_metrics": {
                "test_mse": float(model_info['test_mse']),
                "test_r2": float(model_info['test_r2'])
            },
            "input_features": {
                "demand_index": demand_index,
                "competition_pressure": competition_pressure,
                "operational_cost": operational_cost,
                "seasonality_index": seasonality_index,
                "marketing_intensity": marketing_intensity,
                "time_slot": time_slot,
                "day_of_week": day_of_week
            }
        }), 200
        
    except ValueError as e:
        return jsonify({"error": f"Valeur invalide: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la prédiction: {str(e)}"}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Prédit les prix pour plusieurs ensembles de features
    
    Body JSON attendu:
    {
        "predictions": [
            {features1},
            {features2},
            ...
        ]
    }
    """
    if model_info is None:
        return jsonify({"error": "Modèle non chargé"}), 500
    
    try:
        data = request.get_json()
        
        if "predictions" not in data or not isinstance(data["predictions"], list):
            return jsonify({"error": "Format invalide, 'predictions' doit être une liste"}), 400
        
        results = []
        
        for idx, item in enumerate(data["predictions"]):
            try:
                # Réutilise la logique de prédiction
                time_slot = int(item["time_slot"])
                day_of_week = int(item["day_of_week"])
                
                time_slot_sin = np.sin(2 * np.pi * time_slot / 24)
                time_slot_cos = np.cos(2 * np.pi * time_slot / 24)
                day_sin = np.sin(2 * np.pi * day_of_week / 7)
                day_cos = np.cos(2 * np.pi * day_of_week / 7)
                
                continuous_features = np.array([
                    float(item["demand_index"]),
                    float(item["competition_pressure"]),
                    float(item["operational_cost"]),
                    float(item["seasonality_index"]),
                    float(item["marketing_intensity"])
                ]).reshape(1, -1)
                
                cyclical_features = np.array([
                    time_slot_sin, time_slot_cos, day_sin, day_cos
                ]).reshape(1, -1)
                
                continuous_features_scaled = model_info['scaler'].transform(continuous_features)
                X = np.concatenate([continuous_features_scaled, cyclical_features], axis=1)
                
                predicted_price = model_info['model'].predict(X)[0, 0]
                
                results.append({
                    "index": idx,
                    "predicted_price": float(predicted_price),
                    "input": item
                })
                
            except Exception as e:
                results.append({
                    "index": idx,
                    "error": str(e),
                    "input": item
                })
        
        return jsonify({
            "results": results,
            "model_metrics": {
                "test_mse": float(model_info['test_mse']),
                "test_r2": float(model_info['test_r2'])
            }
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la prédiction batch: {str(e)}"}), 500


# ===================================================
# Lancement de l'application
# ===================================================
if __name__ == '__main__':
    print("\n" + "="*50)
    print(" API Flask - Prédiction de Prix Dynamique")
    print("="*50)
    app.run(debug=True, host='0.0.0.0', port=5000)