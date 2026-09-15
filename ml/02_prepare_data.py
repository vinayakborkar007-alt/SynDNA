import pandas as pd

# Load combined dataset
DATASET_PATH = "dataset/combined_features.csv"

df = pd.read_csv(DATASET_PATH)

# Target column
TARGET = "is_synthetic"

# Metadata columns that should NOT be used for training
METADATA_COLUMNS = [
    "sequence_id",
    "filename",
    "source",
    "organism",
    "taxonomy",
    "sequence_type"
]

# Separate target from dataset
y = df[TARGET]

# Remove target + metadata
X = df.drop(
    columns=METADATA_COLUMNS + [TARGET]
)

# Keep only numerical features
X = X.select_dtypes(include=["number"])

print("=" * 60)
print("SynDNA - Feature Preparation")
print("=" * 60)

print("\nOriginal dataset shape:")
print(df.shape)

print("\nFeature matrix X shape:")
print(X.shape)

print("\nTarget y shape:")
print(y.shape)

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget names:")
print("0 = Natural")
print("1 = Synthetic")

print("\nFirst 5 feature rows:")
print(X.head())

print("\nNumber of features used for ML:")
print(X.shape[1])

print("\nFeature preparation completed.")