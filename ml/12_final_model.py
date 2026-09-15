import pandas as pd
import joblib
from xgboost import XGBClassifier

DATASET_PATH = "dataset/combined_features.csv"

TARGET = "is_synthetic"

METADATA_COLUMNS = [
    "sequence_id",
    "filename",
    "source",
    "organism",
    "taxonomy",
    "sequence_type"
]

df = pd.read_csv(DATASET_PATH)

y = df[TARGET]

X = df.drop(
    columns=METADATA_COLUMNS + [TARGET],
    errors="ignore"
)

X = X.select_dtypes(include=["number"])

selected_features = pd.read_csv(
    "dataset/X_train_reduced.csv"
).columns.tolist()

X = X[selected_features]

print("SynDNA - Final Model Training")
print("=" * 40)

print(f"\nTotal samples : {X.shape[0]}")
print(f"Features      : {X.shape[1]}")

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

print("\nTraining final model...")

model.fit(X, y)

print("\nFinal model training completed!")

joblib.dump(
    model,
    "ml/syndna_final_model.pkl"
)

pd.Series(selected_features).to_csv(
    "ml/final_feature_columns.csv",
    index=False,
    header=["feature"]
)

print("\nSaved:")
print("ml/syndna_final_model.pkl")
print("ml/final_feature_columns.csv")

print("\nFinal model ready!")