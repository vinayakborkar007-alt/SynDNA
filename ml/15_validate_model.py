import os
import pandas as pd
import joblib

MODEL_PATH = "ml/syndna_final_model.pkl"
FEATURES_PATH = "ml/final_feature_columns.csv"
TEST_PATH = "dataset/X_test_reduced.csv"

print("SynDNA - Final Model Validation")
print("=" * 40)

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Model file not found.")
    exit()

if not os.path.exists(FEATURES_PATH):
    print("\nERROR: Feature list not found.")
    exit()

if not os.path.exists(TEST_PATH):
    print("\nERROR: Test dataset not found.")
    exit()

model = joblib.load(MODEL_PATH)

features = pd.read_csv(FEATURES_PATH)["feature"].tolist()
test_data = pd.read_csv(TEST_PATH)

print(f"\nModel loaded        : OK")
print(f"Feature count       : {len(features)}")
print(f"Test feature count  : {len(test_data.columns)}")

if len(features) != 377:
    print("\nERROR: Expected exactly 377 features.")
    exit()

if len(features) != len(set(features)):
    print("\nERROR: Duplicate features detected.")
    exit()

missing_features = [
    feature for feature in features
    if feature not in test_data.columns
]

extra_features = [
    feature for feature in test_data.columns
    if feature not in features
]

if missing_features:
    print("\nERROR: Missing features:")
    print(missing_features)
    exit()

if extra_features:
    print("\nWARNING: Extra features detected:")
    print(extra_features)

X_test = test_data[features]

prediction = model.predict(X_test)

print("\nPrediction test      : OK")
print(f"Predictions generated: {len(prediction)}")

print("\nFinal validation     : PASSED")
print("\nModel is ready for backend integration.")