import pandas as pd
import numpy as np

X_train = pd.read_csv("dataset/X_train_filtered.csv")
X_test = pd.read_csv("dataset/X_test_filtered.csv")

print("SynDNA - Correlation Filtering")
print("=" * 40)

print(f"\nOriginal features : {X_train.shape[1]}")

correlation_matrix = X_train.corr().abs()

upper_triangle = correlation_matrix.where(
    np.triu(
        np.ones(correlation_matrix.shape),
        k=1
    ).astype(bool)
)

threshold = 0.95

features_to_remove = [
    column
    for column in upper_triangle.columns
    if any(upper_triangle[column] > threshold)
]

X_train_reduced = X_train.drop(
    columns=features_to_remove
)

X_test_reduced = X_test.drop(
    columns=features_to_remove
)

print(f"\nCorrelation threshold : {threshold}")
print(f"Highly correlated features removed : {len(features_to_remove)}")
print(f"Features remaining : {X_train_reduced.shape[1]}")

X_train_reduced.to_csv(
    "dataset/X_train_reduced.csv",
    index=False
)

X_test_reduced.to_csv(
    "dataset/X_test_reduced.csv",
    index=False
)

print("\nSaved:")
print("X_train_reduced.csv")
print("X_test_reduced.csv")

print("\nCorrelation filtering completed successfully!")
