import pandas as pd
import numpy as np

# ============================================================
# FinSecure AI
# Stage 1 - PaySim Feature Validation
# ============================================================

print("\nLoading engineered PaySim dataset...")

# Path to engineered dataset
file_path = "dataset/paysim_engineered.csv"

# Load dataset
df = pd.read_csv(file_path)

print("Dataset loaded successfully.")

print("\n========== DATASET INFORMATION ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ============================================================
# 1. COLUMN INFORMATION
# ============================================================

print("\n========== COLUMN INFORMATION ==========")

for column in df.columns:
    print(column)


# ============================================================
# 2. MISSING VALUES
# ============================================================

print("\n========== MISSING VALUE CHECK ==========")

missing_values = df.isnull().sum()

print(missing_values)

total_missing = missing_values.sum()

if total_missing == 0:
    print("\nResult: No missing values found.")
else:
    print("\nWARNING: Missing values found:", total_missing)


# ============================================================
# 3. INFINITE VALUES
# ============================================================

print("\n========== INFINITE VALUE CHECK ==========")

numeric_columns = df.select_dtypes(include=np.number).columns

infinite_values = np.isinf(df[numeric_columns]).sum()

print(infinite_values)

total_infinite = infinite_values.sum()

if total_infinite == 0:
    print("\nResult: No infinite values found.")
else:
    print("\nWARNING: Infinite values found:", total_infinite)


# ============================================================
# 4. DUPLICATE TRANSACTIONS
# ============================================================

print("\n========== DUPLICATE CHECK ==========")

duplicate_count = df.duplicated().sum()

print("Duplicate rows:", duplicate_count)

if duplicate_count == 0:
    print("Result: No duplicate rows found.")
else:
    print("WARNING: Duplicate rows found.")


# ============================================================
# 5. TARGET DISTRIBUTION
# ============================================================

print("\n========== FRAUD TARGET DISTRIBUTION ==========")

fraud_distribution = df["isFraud"].value_counts()

print(fraud_distribution)

print("\nFraud Percentage:")

fraud_percentage = df["isFraud"].value_counts(normalize=True) * 100

print(fraud_percentage)


# ============================================================
# 6. NUMERIC FEATURE STATISTICS
# ============================================================

print("\n========== NUMERIC FEATURE STATISTICS ==========")

print(df[numeric_columns].describe().T)


# ============================================================
# 7. ENGINEERED FEATURE VALIDATION
# ============================================================

engineered_features = [
    "sender_balance_change",
    "sender_balance_change_ratio",
    "receiver_balance_change",
    "amount_to_sender_balance_ratio",
    "expected_sender_balance",
    "sender_balance_error",
    "expected_receiver_balance",
    "receiver_balance_error"
]

print("\n========== ENGINEERED FEATURE VALIDATION ==========")

for feature in engineered_features:

    if feature not in df.columns:
        print("\nMissing feature:", feature)
        continue

    print("\n----------------------------------------")
    print("Feature:", feature)

    print("Minimum:", df[feature].min())
    print("Maximum:", df[feature].max())
    print("Mean:", df[feature].mean())
    print("Median:", df[feature].median())


# ============================================================
# 8. ZERO VALUE ANALYSIS
# ============================================================

print("\n========== ZERO VALUE ANALYSIS ==========")

for feature in engineered_features:

    if feature in df.columns:

        zero_count = (df[feature] == 0).sum()

        zero_percentage = (zero_count / len(df)) * 100

        print(
            f"{feature}: "
            f"{zero_count} zeros "
            f"({zero_percentage:.2f}%)"
        )


# ============================================================
# 9. NEGATIVE VALUE ANALYSIS
# ============================================================

print("\n========== NEGATIVE VALUE ANALYSIS ==========")

for feature in engineered_features:

    if feature in df.columns:

        negative_count = (df[feature] < 0).sum()

        negative_percentage = (
            negative_count / len(df)
        ) * 100

        print(
            f"{feature}: "
            f"{negative_count} negative values "
            f"({negative_percentage:.2f}%)"
        )


# ============================================================
# 10. OUTLIER ANALYSIS USING IQR
# ============================================================

print("\n========== OUTLIER ANALYSIS ==========")

for feature in engineered_features:

    if feature not in df.columns:
        continue

    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[feature] < lower_bound) |
        (df[feature] > upper_bound)
    ]

    print(
        f"{feature}: "
        f"{len(outliers)} outliers"
    )


# ============================================================
# 11. FRAUD VS LEGITIMATE FEATURE COMPARISON
# ============================================================

print("\n========== FRAUD VS LEGITIMATE FEATURE COMPARISON ==========")

for feature in engineered_features:

    if feature not in df.columns:
        continue

    comparison = df.groupby("isFraud")[feature].agg(
        ["mean", "median", "min", "max"]
    )

    print("\n----------------------------------------")
    print("Feature:", feature)
    print(comparison)


# ============================================================
# 12. CORRELATION WITH FRAUD
# ============================================================

print("\n========== FEATURE CORRELATION WITH FRAUD ==========")

correlation = (
    df[numeric_columns]
    .corr()["isFraud"]
    .sort_values(ascending=False)
)

print(correlation)


# ============================================================
# 13. HIGH CORRELATION BETWEEN FEATURES
# ============================================================

print("\n========== HIGH FEATURE CORRELATION ==========")

correlation_matrix = df[numeric_columns].corr()

high_correlation_found = False

for i in range(len(correlation_matrix.columns)):

    for j in range(i + 1, len(correlation_matrix.columns)):

        correlation_value = correlation_matrix.iloc[i, j]

        if abs(correlation_value) >= 0.90:

            feature_1 = correlation_matrix.columns[i]
            feature_2 = correlation_matrix.columns[j]

            print(
                f"{feature_1} <-> {feature_2}: "
                f"{correlation_value:.4f}"
            )

            high_correlation_found = True


if not high_correlation_found:

    print("No feature pairs with correlation >= 0.90 found.")


# ============================================================
# 14. FRAUD TRANSACTION TYPE ANALYSIS
# ============================================================

print("\n========== FRAUD BY TRANSACTION TYPE ==========")

fraud_by_type = (
    df.groupby("type")["isFraud"]
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


# ============================================================
# 15. FINAL VALIDATION SUMMARY
# ============================================================

print("\n========================================")
print("FEATURE VALIDATION COMPLETE")
print("========================================")

print("\nDataset:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nMissing values:", total_missing)
print("Infinite values:", total_infinite)
print("Duplicate rows:", duplicate_count)

print("\nEngineered features checked:", len(engineered_features))

print("\nNext step:")
print("Feature selection and model preparation.")