"""Unit tests for the FFNN model architecture and predictions."""

import tempfile
import unittest
from pathlib import Path
import numpy as np

from src.train import NumpyFFNN


class TestModelArchitecture(unittest.TestCase):

    def setUp(self):
        self.model = NumpyFFNN(input_dim=5, hidden_dim1=16, hidden_dim2=8, dropout_rate=0.20, random_seed=42)

    def test_input_and_output_shape(self):
        sample = np.array([[1.0, 2.0, 800.0, 3.0, 150.0]])
        self.assertEqual(sample.shape, (1, 5))

        proba = self.model.predict_proba(sample)
        self.assertEqual(proba.shape, (1,))
        self.assertTrue(0.0 <= proba[0] <= 1.0)

        batch = np.random.randn(10, 5)
        batch_probs = self.model.predict_proba(batch)
        self.assertEqual(batch_probs.shape, (10,))
        self.assertTrue(np.all((batch_probs >= 0.0) & (batch_probs <= 1.0)))

    def test_binary_prediction_threshold(self):
        sample = np.random.randn(5, 5)
        preds_50 = self.model.predict(sample, threshold=0.50)
        preds_85 = self.model.predict(sample, threshold=0.85)

        self.assertEqual(preds_50.shape, (5,))
        self.assertEqual(preds_85.shape, (5,))
        self.assertTrue(set(preds_50).issubset({0, 1}))
        self.assertTrue(set(preds_85).issubset({0, 1}))

    def test_save_and_load_consistency(self):
        sample = np.random.randn(4, 5)
        orig_preds = self.model.predict_proba(sample)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "test_model.keras"
            self.model.save(model_path)
            self.assertTrue(model_path.exists())

            loaded_model = NumpyFFNN.load(model_path)
            loaded_preds = loaded_model.predict_proba(sample)

            np.testing.assert_allclose(orig_preds, loaded_preds, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
