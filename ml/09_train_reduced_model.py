import pandas as pd
import joblib
from xgboost import XGBClassifier

X_train = pd.read_csv("dataset/X_train_reduced.csv")
y_train = pd.read_csv("dataset/y_train.csv").squeeze()

print("SynDNA - Reduced Feature XGBoost Training")
print("=" * 40)

print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Features         : {X_train.shape[1]}")

model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

print("\nTraining model...")

model.fit(X_train, y_train)

print("\nModel training completed!")

joblib.dump(
    model,
    "ml/xgboost_reduced_model.pkl"
)

print("\nModel saved:")
print("ml/xgboost_reduced_model.pkl")