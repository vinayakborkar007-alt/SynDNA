import pandas as pd
import joblib

MODEL_PATH = "ml/syndna_final_model.pkl"
FEATURES_PATH = "ml/final_feature_columns.csv"

model = joblib.load(MODEL_PATH)

feature_columns = pd.read_csv(
    FEATURES_PATH
)["feature"].tolist()

print("SynDNA - ML Prediction")
print("=" * 40)

print(f"\nExpected features: {len(feature_columns)}")

input_path = input(
    "\nEnter path to feature CSV: "
).strip()

data = pd.read_csv(input_path)

missing_features = [
    feature
    for feature in feature_columns
    if feature not in data.columns
]

if missing_features:
    print("\nERROR: Missing required features:")
    print(missing_features)
    exit()

X = data[feature_columns]

predictions = model.predict(X)
probabilities = model.predict_proba(X)

for i in range(len(data)):
    prediction = predictions[i]

    natural_probability = probabilities[i][0]
    synthetic_probability = probabilities[i][1]

    if prediction == 1:
        result = "Synthetic"
    else:
        result = "Natural"

    print("\nSequence", i + 1)
    print("-" * 30)
    print(f"Prediction           : {result}")
    print(f"Natural probability  : {natural_probability:.4f}")
    print(f"Synthetic probability: {synthetic_probability:.4f}")