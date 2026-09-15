import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
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

print("SynDNA - 5-Fold Cross-Validation")
print("=" * 40)

print("\nDataset shape:")
print(X.shape)

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

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = [
    "accuracy",
    "precision",
    "recall",
    "f1"
]

results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=scoring
)

print("\nCross-Validation Results")
print("=" * 40)

for metric in scoring:
    scores = results[f"test_{metric}"]

    print(f"\n{metric.capitalize()}:")
    print("Fold scores:", [round(score, 4) for score in scores])
    print(f"Mean       : {scores.mean():.4f}")
    print(f"Std        : {scores.std():.4f}")

print("\nCross-validation completed successfully!")