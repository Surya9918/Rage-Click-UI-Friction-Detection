"""Integration tests for FastAPI endpoints."""

import sys
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.server import app
from src.preprocessing import DataPreprocessor
from src.train import NumpyFFNN
import numpy as np


class TestFastAPIEndpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        models_dir = PROJECT_ROOT / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "friction_model.keras"
        scaler_path = models_dir / "scaler.joblib"

        if not model_path.exists() or not scaler_path.exists():
            model = NumpyFFNN(input_dim=5, hidden_dim1=16, hidden_dim2=8)
            model.save(model_path)
            preprocessor = DataPreprocessor()
            X_dummy = np.random.randn(20, 5)
            preprocessor.fit(X_dummy)
            preprocessor.save(scaler_path)

        cls.client = TestClient(app)

    def test_get_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))

    def test_get_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["friction_threshold"], 0.85)

    def test_post_predict_valid_features(self):
        payload = {
            "click_frequency": 5.0,
            "rapid_fire_clicks": 4.0,
            "maximum_cursor_velocity": 1500.0,
            "erratic_direction_changes": 5.0,
            "scroll_thrashing": 800.0,
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("friction_probability", data)
        self.assertIn("friction_detected", data)
        self.assertIn("threshold", data)
        self.assertIn("features", data)
        self.assertTrue(0.0 <= data["friction_probability"] <= 1.0)
        self.assertEqual(data["threshold"], 0.85)

    def test_post_predict_invalid_payload(self):
        response = self.client.post("/predict", json={"click_frequency": -1.0})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
