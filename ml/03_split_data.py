import pandas as pd
from sklearn.model_selection import train_test_split

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

print("SynDNA - Train/Test Split")
print("=" * 40)

print("\nOriginal dataset:")
print(df.shape)

y = df[TARGET]

X = df.drop(
    columns=METADATA_COLUMNS + [TARGET],
    errors="ignore"
)

X = X.select_dtypes(include=["number"])

print("\nFeature matrix X:")
print(X.shape)

print("\nTarget y:")
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nSplit Results")
print("=" * 40)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

X_train.to_csv("dataset/X_train.csv", index=False)
X_test.to_csv("dataset/X_test.csv", index=False)

y_train.to_csv("dataset/y_train.csv", index=False)
y_test.to_csv("dataset/y_test.csv", index=False)

print("\nSaved files:")
print("X_train.csv")
print("X_test.csv")
print("y_train.csv")
print("y_test.csv")

print("\nTrain/test split completed successfully!")
