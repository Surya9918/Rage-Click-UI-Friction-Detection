import pandas as pd

class Labeler:
    def label_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        
        c1 = df_copy["rapid_fire_clicks"] >= 3
        c2 = (df_copy["erratic_direction_changes"] >= 4) & (df_copy["maximum_cursor_velocity"] >= 6591.0)
        c3 = df_copy["scroll_thrashing"] >= 2000.0
        c4 = (df_copy["click_frequency"] >= 2.0) & (df_copy["rapid_fire_clicks"] >= 2)
        
        import numpy as np
        friction_mask = c1 | c2 | c3 | c4
        labels = friction_mask.astype(int).values.copy()
        
        # Add 8% random label noise to simulate real human unpredictability
        # (Users doing strange things but not frustrated, or frustrated users hiding it well)
        np.random.seed(42)
        noise_mask = np.random.rand(len(labels)) < 0.08
        labels[noise_mask] = 1 - labels[noise_mask]
        
        df_copy["label"] = labels
        
        return df_copy
