import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.train import train
from src.evaluate import evaluate
from src.data_generation import SyntheticTelemetryGenerator
from src.labeling import Labeler

def run_pipeline():
    data_dir = PROJECT_ROOT / "data" / "synthetic"
    models_dir = PROJECT_ROOT / "models"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    
    data_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    features_csv = data_dir / "synthetic_features.csv"
    if not features_csv.exists():
        generator = SyntheticTelemetryGenerator()
        generator.save_synthetic_dataset(data_dir)

    df = pd.read_csv(features_csv)
    
    labeler = Labeler()
    df_labeled = labeler.label_dataframe(df)
    df_labeled.to_csv(processed_dir / "labeled_features.csv", index=False)

    model, scaler, X_test, y_test = train(df_labeled, models_dir)
    
    evaluate(model, X_test, y_test, models_dir)

if __name__ == "__main__":
    run_pipeline()
