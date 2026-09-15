import pandas as pd
import joblib

MODEL_PATH = "ml/syndna_final_model.pkl"
FEATURES_PATH = "ml/final_feature_columns.csv"

model = joblib.load(MODEL_PATH)

features = pd.read_csv(FEATURES_PATH)

print("SynDNA - Final ML Model Information")
print("=" * 40)

print(f"\nModel type        : {type(model).__name__}")
print(f"Number of features: {len(features)}")
print(f"Number of trees   : {model.n_estimators}")
print(f"Max depth         : {model.max_depth}")
print(f"Learning rate     : {model.learning_rate}")
print(f"Subsample         : {model.subsample}")
print(f"Feature sampling  : {model.colsample_bytree}")

print("\nClasses:")
print("0 = Natural")
print("1 = Synthetic")

print("\nModel file:")
print(MODEL_PATH)

print("\nFeature file:")
print(FEATURES_PATH)

print("\nTop 20 features:")
for i, feature in enumerate(features["feature"].head(20), start=1):
    print(f"{i:2}. {feature}")