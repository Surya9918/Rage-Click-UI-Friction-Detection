import pandas as pd
import numpy as np
from pathlib import Path
import random

class SyntheticTelemetryGenerator:
    def save_synthetic_dataset(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        n_samples = 1000
        n_friction = 100
        n_normal = n_samples - n_friction
        
        rows = []
        for i in range(n_normal):
            rows.append({
                "click_frequency": max(0.0, np.random.exponential(scale=0.35)),
                "rapid_fire_clicks": np.random.choice([0, 1, 2], p=[0.90, 0.08, 0.02]),
                "maximum_cursor_velocity": np.random.lognormal(mean=5.8, sigma=0.45),
                "erratic_direction_changes": np.random.choice([0, 1, 2, 3], p=[0.70, 0.20, 0.08, 0.02]),
                "scroll_thrashing": 0.0 if random.random() > 0.05 else np.random.uniform(20, 150),
            })
            
        for i in range(n_friction):
            rows.append({
                "click_frequency": np.random.uniform(1.8, 4.5),
                "rapid_fire_clicks": np.random.randint(3, 10),
                "maximum_cursor_velocity": np.random.uniform(6600, 10000),
                "erratic_direction_changes": np.random.choice([1, 2, 3, 5], p=[0.3, 0.3, 0.25, 0.15]),
                "scroll_thrashing": 0.0 if random.random() > 0.15 else np.random.uniform(50, 300),
            })
            
        df = pd.DataFrame(rows)
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        features_csv_path = output_dir / "synthetic_features.csv"
        df.to_csv(features_csv_path, index=False)
        return features_csv_path
