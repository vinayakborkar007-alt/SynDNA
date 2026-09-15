import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

X_test = pd.read_csv("dataset/X_test_reduced.csv")
y_test = pd.read_csv("dataset/y_test.csv").squeeze()

model = joblib.load("ml/xgboost_reduced_model.pkl")

print("SynDNA - Reduced Model Evaluation")
print("=" * 40)

print(f"\nTest samples : {X_test.shape[0]}")
print(f"Features     : {X_test.shape[1]}")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)

print("\nEvaluation Results")
print("=" * 40)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Natural", "Synthetic"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nReduced model evaluation completed!")