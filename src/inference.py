import torch
import joblib
import numpy as np
from pathlib import Path
from src.train import FrictionFFNN

def predict(features_dict, models_dir="models"):
    models_dir = Path(models_dir)
    checkpoint = torch.load(models_dir / "friction_model.pt")
    
    model = FrictionFFNN(input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    scaler = joblib.load(models_dir / "scaler.joblib")
    
    feature_names = ["click_frequency", "rapid_fire_clicks", "maximum_cursor_velocity", "erratic_direction_changes", "scroll_thrashing"]
    ordered_features = [features_dict.get(k, 0.0) for k in feature_names]
    
    X = np.array([ordered_features])
    X_scaled = scaler.transform(X)
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        base_prob = model(X_tensor).item()
        
        importances = {}
        for idx, name in enumerate(feature_names):
            perturbed = X_scaled.copy()
            perturbed[0, idx] = 0.0
            perturbed_tensor = torch.FloatTensor(perturbed)
            perturbed_prob = model(perturbed_tensor).item()
            impact = max(0.0, base_prob - perturbed_prob)
            importances[name] = impact
            
        total_impact = sum(importances.values())
        if total_impact > 1e-6:
            feature_importance = {k: round(v / total_impact, 4) for k, v in importances.items()}
        else:
            feature_importance = {k: 0.20 for k in feature_names}
        
    return {
        "friction_probability": round(base_prob, 4),
        "friction_detected": bool(base_prob >= 0.85),
        "feature_importance": feature_importance
    }
