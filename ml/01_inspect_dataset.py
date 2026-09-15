import pandas as pd
import os

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"C:\Users\Ronak Jain\OneDrive\Desktop\Project\SynDNA\dataset\combined_features.csv"


# ============================================================
# LOAD DATASET
# ============================================================

print("\n" + "=" * 60)
print("SynDNA - Dataset Inspection")
print("=" * 60)

if not os.path.exists(DATASET_PATH):
    print(f"\nERROR: Dataset not found at:")
    print(DATASET_PATH)
    print("\nChange DATASET_PATH in this file to your actual CSV path.")
    exit()

df = pd.read_csv(DATASET_PATH)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n[1] Dataset Shape")
print("-" * 60)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# COLUMN NAMES
# ============================================================

print("\n[2] Columns")
print("-" * 60)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:3}. {column}")


# ============================================================
# DATA TYPES
# ============================================================

print("\n[3] Data Types")
print("-" * 60)

print(df.dtypes)


# ============================================================
# FIRST 5 ROWS
# ============================================================

print("\n[4] First 5 Rows")
print("-" * 60)

print(df.head().to_string())


# ============================================================
# MISSING VALUES
# ============================================================

print("\n[5] Missing Values")
print("-" * 60)

missing = df.isnull().sum()

missing_table = pd.DataFrame({
    "column": missing.index,
    "missing_values": missing.values,
    "missing_percentage": (
        missing.values / len(df) * 100
    ).round(2)
})

print(
    missing_table[
        missing_table["missing_values"] > 0
    ].to_string(index=False)
)

if missing.sum() == 0:
    print("No missing values found.")


# ============================================================
# DUPLICATES
# ============================================================

print("\n[6] Duplicate Rows")
print("-" * 60)

duplicates = df.duplicated().sum()

print(f"Duplicate rows: {duplicates}")


# ============================================================
# NUMERICAL FEATURES
# ============================================================

print("\n[7] Numerical Features")
print("-" * 60)

numeric_columns = df.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns

print(f"Number of numerical columns: {len(numeric_columns)}")

for column in numeric_columns:
    print(f" - {column}")


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

print("\n[8] Categorical / Text Features")
print("-" * 60)

categorical_columns = df.select_dtypes(
    include=["object", "category", "bool"]
).columns

print(f"Number of categorical/text columns: {len(categorical_columns)}")

for column in categorical_columns:
    print(f" - {column}")


# ============================================================
# UNIQUE VALUES
# ============================================================

print("\n[9] Unique Values")
print("-" * 60)

for column in df.columns:

    unique_count = df[column].nunique()

    print(
        f"{column:<35} "
        f"{unique_count:>8} unique values"
    )


# ============================================================
# NUMERICAL STATISTICS
# ============================================================

print("\n[10] Numerical Statistics")
print("-" * 60)

if len(numeric_columns) > 0:
    print(df[numeric_columns].describe().T.to_string())
else:
    print("No numerical columns found.")


# ============================================================
# POSSIBLE TARGET COLUMNS
# ============================================================

print("\n[11] Possible Target Columns")
print("-" * 60)

keywords = [
    "label",
    "target",
    "class",
    "synthetic",
    "is_synthetic",
    "y"
]

possible_targets = []

for column in df.columns:

    column_lower = column.lower()

    for keyword in keywords:

        if keyword in column_lower:
            possible_targets.append(column)
            break


if possible_targets:

    for column in possible_targets:

        print(f"\nTarget candidate: {column}")

        print(
            df[column]
            .value_counts(dropna=False)
            .to_string()
        )

else:

    print("No obvious target column detected.")


# ============================================================
# END
# ============================================================

print("\n" + "=" * 60)
print("Dataset inspection completed.")
print("=" * 60)
