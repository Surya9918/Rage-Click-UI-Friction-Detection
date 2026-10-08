from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch

import json
from pathlib import Path

def evaluate(model, X_test, y_test, models_dir=None):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test)
        probs = model(X_tensor).numpy().ravel()
        preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    report = classification_report(y_test, preds)
    print("Classification Report:\n", report)
    
    if models_dir is not None:
        stats = {
            "accuracy": float(acc),
            "report_dict": classification_report(y_test, preds, output_dict=True)
        }
        with open(Path(models_dir) / "metadata.json", "w") as f:
            json.dump(stats, f, indent=4)
