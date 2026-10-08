import pandas as pd
import numpy as np
from pathlib import Path
import random

class SyntheticTelemetryGenerator:
    def save_synthetic_dataset(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        n_samples = 25000
        n_friction = 3500
        n_normal = n_samples - n_friction
        
        rows = []
        for i in range(n_normal):
            rows.append({
                "click_frequency": max(0.0, np.random.exponential(scale=0.6)),
                "rapid_fire_clicks": np.random.choice([0, 1, 2, 3, 4], p=[0.85, 0.08, 0.04, 0.02, 0.01]),
                "maximum_cursor_velocity": np.random.lognormal(mean=6.5, sigma=0.8),
                "erratic_direction_changes": np.random.choice([0, 1, 2, 3, 4, 5], p=[0.60, 0.20, 0.10, 0.05, 0.03, 0.02]),
                "scroll_thrashing": 0.0 if random.random() > 0.1 else np.random.uniform(50, 2500),
            })
            
        for i in range(n_friction):
            rows.append({
                "click_frequency": max(0.0, np.random.normal(loc=2.2, scale=1.2)),
                "rapid_fire_clicks": np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], p=[0.05, 0.10, 0.15, 0.25, 0.20, 0.15, 0.05, 0.05]),
                "maximum_cursor_velocity": max(0.0, np.random.normal(loc=6000, scale=2500)),
                "erratic_direction_changes": np.random.choice([1, 2, 3, 4, 5, 6], p=[0.1, 0.15, 0.25, 0.25, 0.15, 0.1]),
                "scroll_thrashing": 0.0 if random.random() > 0.35 else max(0.0, np.random.normal(loc=2200, scale=1200)),
            })
            
        df = pd.DataFrame(rows)
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        features_csv_path = output_dir / "synthetic_features.csv"
        df.to_csv(features_csv_path, index=False)
        return features_csv_path
