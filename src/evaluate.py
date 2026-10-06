from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch

def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test)
        probs = model(X_tensor).numpy().ravel()
        preds = (probs >= 0.5).astype(int)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))
