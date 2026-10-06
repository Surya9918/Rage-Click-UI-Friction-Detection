import sys
from pathlib import Path
import pandas as pd
import torch
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import evaluate
from src.train import FrictionFFNN

def main():
    models_dir = PROJECT_ROOT / "models"
    data_path = PROJECT_ROOT / "data" / "processed" / "labeled_features.csv"

    checkpoint = torch.load(models_dir / "friction_model.pt")
    model = FrictionFFNN(input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    
    scaler = joblib.load(models_dir / "scaler.joblib")
    
    df = pd.read_csv(data_path)
    features = ["click_frequency", "rapid_fire_clicks", "maximum_cursor_velocity", "erratic_direction_changes", "scroll_thrashing"]
    
    X = scaler.transform(df[features].values)
    y = df["label"].values

    evaluate(model, X, y)

if __name__ == "__main__":
    main()
