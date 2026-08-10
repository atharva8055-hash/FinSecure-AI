import pandas as pd

# ============================================================
# FinSecure AI
# Module 1: AI-Based UPI Fraud Detection
#
# Stage 1: PaySim Dataset
# Feature Selection and ML Dataset Preparation
# ============================================================


# ------------------------------------------------------------
# 1. Load engineered dataset
# ------------------------------------------------------------

print("\nLoading engineered PaySim dataset...")

file_path = "dataset/paysim_engineered.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully.")
print("Original rows:", df.shape[0])
print("Original columns:", df.shape[1])


# ------------------------------------------------------------
# 2. Define target variable
# ------------------------------------------------------------

target = "isFraud"


# ------------------------------------------------------------
# 3. Features selected for the first ML baseline
# ------------------------------------------------------------

selected_features = [
    "step",
    "type",
    "amount",

    "oldbalanceOrg",
    "newbalanceOrig",

    "oldbalanceDest",
    "newbalanceDest",

    "sender_balance_change",
    "sender_balance_change_ratio",

    "receiver_balance_change",

    "amount_to_sender_balance_ratio",

    "expected_sender_balance",
    "sender_balance_error",

    "expected_receiver_balance",
    "receiver_balance_error"
]


# ------------------------------------------------------------
# 4. Check that all selected features exist
# ------------------------------------------------------------

print("\n========== FEATURE AVAILABILITY CHECK ==========")

missing_features = [
    feature
    for feature in selected_features
    if feature not in df.columns
]

if missing_features:

    print("ERROR: The following features were not found:")

    for feature in missing_features:
        print("-", feature)

    raise ValueError("Some selected features are missing from the dataset.")

else:

    print("All selected features are available.")


# ------------------------------------------------------------
# 5. Create ML dataset
# ------------------------------------------------------------

ml_columns = selected_features + [target]

ml_df = df[ml_columns].copy()


# ------------------------------------------------------------
# 6. Display selected features
# ------------------------------------------------------------

print("\n========== SELECTED FEATURES ==========")

for feature in selected_features:
    print(feature)

print("\nTarget variable:")
print(target)


# ------------------------------------------------------------
# 7. Display feature count
# ------------------------------------------------------------

print("\n========== FEATURE COUNT ==========")

print("Number of input features:", len(selected_features))
print("Target column:", target)
print("Total columns:", len(ml_columns))


# ------------------------------------------------------------
# 8. Check missing values
# ------------------------------------------------------------

print("\n========== MISSING VALUE CHECK ==========")

missing_values = ml_df.isnull().sum()

print(missing_values)

if missing_values.sum() == 0:

    print("\nResult: No missing values found.")

else:

    print("\nWARNING: Missing values detected.")


# ------------------------------------------------------------
# 9. Check infinite values in numerical columns
# ------------------------------------------------------------

print("\n========== INFINITE VALUE CHECK ==========")

numeric_columns = ml_df.select_dtypes(
    include=["int64", "float64"]
).columns

infinite_values = (
    ml_df[numeric_columns]
    .isin([float("inf"), float("-inf")])
    .sum()
)

print(infinite_values)

if infinite_values.sum() == 0:

    print("\nResult: No infinite values found.")

else:

    print("\nWARNING: Infinite values detected.")


# ------------------------------------------------------------
# 10. Transaction type analysis
# ------------------------------------------------------------

print("\n========== TRANSACTION TYPE ==========")

print(ml_df["type"].value_counts())


# ------------------------------------------------------------
# 11. Fraud distribution
# ------------------------------------------------------------

print("\n========== FRAUD TARGET DISTRIBUTION ==========")

fraud_distribution = ml_df[target].value_counts()

print(fraud_distribution)

print("\nFraud percentage:")

print(
    ml_df[target]
    .value_counts(normalize=True)
    .mul(100)
)


# ------------------------------------------------------------
# 12. Fraud percentage by transaction type
# ------------------------------------------------------------

print("\n========== FRAUD BY TRANSACTION TYPE ==========")

fraud_by_type = (
    ml_df.groupby("type")[target]
    .agg(
        total_transactions="count",
        fraudulent_transactions="sum"
    )
)

fraud_by_type["fraud_percentage"] = (
    fraud_by_type["fraudulent_transactions"]
    / fraud_by_type["total_transactions"]
) * 100

print(fraud_by_type)


# ------------------------------------------------------------
# 13. Numerical feature summary
# ------------------------------------------------------------

print("\n========== NUMERICAL FEATURE SUMMARY ==========")

print(
    ml_df[numeric_columns]
    .describe()
)


# ------------------------------------------------------------
# 14. Prepare categorical transaction type
# ------------------------------------------------------------

print("\n========== TRANSACTION TYPE ENCODING ==========")

print("Original transaction types:")

print(
    ml_df["type"]
    .unique()
)


# One-hot encode transaction type
#
# Example:
# CASH_OUT -> type_CASH_OUT = 1
# TRANSFER -> type_TRANSFER = 1
#
# This prevents the model from treating transaction types
# as numerical values.

ml_df = pd.get_dummies(
    ml_df,
    columns=["type"],
    prefix="type",
    dtype=int
)


print("\nEncoded transaction type columns:")

encoded_type_columns = [
    column
    for column in ml_df.columns
    if column.startswith("type_")
]

for column in encoded_type_columns:
    print(column)


# ------------------------------------------------------------
# 15. Separate input features and target
# ------------------------------------------------------------

X = ml_df.drop(columns=[target])

y = ml_df[target]


print("\n========== ML DATASET ==========")

print("X shape:", X.shape)
print("y shape:", y.shape)


# ------------------------------------------------------------
# 16. Display final ML feature list
# ------------------------------------------------------------

print("\n========== FINAL ML FEATURES ==========")

for index, feature in enumerate(X.columns, start=1):

    print(index, "->", feature)


# ------------------------------------------------------------
# 17. Save prepared dataset
# ------------------------------------------------------------

output_file = "dataset/paysim_ml_dataset.csv"

print("\nSaving prepared ML dataset...")

ml_df.to_csv(
    output_file,
    index=False
)

print("\n========== FEATURE SELECTION COMPLETE ==========")

print("Output file:")
print(output_file)

print("Rows:", ml_df.shape[0])
print("Columns:", ml_df.shape[1])

print("\nNext step:")
print("Train/Test Split and Class Imbalance Handling.")