import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
from pathlib import Path

class FrictionFFNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def train(df: pd.DataFrame, models_dir: Path):
    models_dir.mkdir(parents=True, exist_ok=True)
    features = ["click_frequency", "rapid_fire_clicks", "maximum_cursor_velocity", "erratic_direction_changes", "scroll_thrashing"]
    
    X = df[features].values
    y = df["label"].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, models_dir / "scaler.joblib")
    
    model = FrictionFFNN(input_dim=len(features))
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    
    # Simple training loop
    for epoch in range(100):
        optimizer.zero_grad()
        loss = criterion(model(X_tensor), y_tensor)
        loss.backward()
        optimizer.step()
        
    torch.save({"model_state_dict": model.state_dict(), "input_dim": len(features)}, models_dir / "friction_model.pt")
    return model, scaler, X_test, y_test

