import pandas as pd
from sklearn.feature_selection import VarianceThreshold

X_train = pd.read_csv("dataset/X_train.csv")
X_test = pd.read_csv("dataset/X_test.csv")

print("SynDNA - Feature Filtering")
print("=" * 40)

print(f"\nOriginal training features : {X_train.shape[1]}")
print(f"Original testing features  : {X_test.shape[1]}")

selector = VarianceThreshold(threshold=0)

X_train_filtered = selector.fit_transform(X_train)
X_test_filtered = selector.transform(X_test)

selected_features = X_train.columns[selector.get_support()]

X_train_filtered = pd.DataFrame(
    X_train_filtered,
    columns=selected_features
)

X_test_filtered = pd.DataFrame(
    X_test_filtered,
    columns=selected_features
)

removed_features = X_train.shape[1] - X_train_filtered.shape[1]

print(f"\nConstant features removed : {removed_features}")
print(f"Features remaining        : {X_train_filtered.shape[1]}")

X_train_filtered.to_csv(
    "dataset/X_train_filtered.csv",
    index=False
)

X_test_filtered.to_csv(
    "dataset/X_test_filtered.csv",
    index=False
)

print("\nSaved:")
print("X_train_filtered.csv")
print("X_test_filtered.csv")

print("\nFeature filtering completed successfully!")