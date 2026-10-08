import pandas as pd

class Labeler:
    def label_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        
        c1 = df_copy["rapid_fire_clicks"] >= 3
        c2 = (df_copy["erratic_direction_changes"] >= 4) & (df_copy["maximum_cursor_velocity"] >= 6591.0)
        c3 = df_copy["scroll_thrashing"] >= 400.0
        c4 = (df_copy["click_frequency"] >= 2.0) & (df_copy["rapid_fire_clicks"] >= 2)
        
        friction_mask = c1 | c2 | c3 | c4
        df_copy["label"] = friction_mask.astype(int)
        
        return df_copy
