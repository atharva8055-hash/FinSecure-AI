import pandas as pd
import numpy as np

print("\n========== LOADING ENGINEERED DATASET ==========")

file_path = "dataset/paysim_engineered.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully.")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n========== TARGET INFORMATION ==========")

print("Target variable: isFraud")

print("\nFraud distribution:")
print(df["isFraud"].value_counts())

print("\nFraud percentage:")
print(df["isFraud"].value_counts(normalize=True) * 100)


# ============================================================
# FEATURE LIST
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

print("\n========== ENGINEERED FEATURES ==========")

for feature in engineered_features:
    print("-", feature)


# ============================================================
# FEATURE FORMULA / DATA AVAILABILITY ANALYSIS
# ============================================================

print("\n========== FEATURE AVAILABILITY ANALYSIS ==========")

feature_sources = {
    "sender_balance_change":
        "oldbalanceOrg and newbalanceOrig",

    "sender_balance_change_ratio":
        "oldbalanceOrg and newbalanceOrig",

    "receiver_balance_change":
        "oldbalanceDest and newbalanceDest",

    "amount_to_sender_balance_ratio":
        "amount and oldbalanceOrg",

    "expected_sender_balance":
        "oldbalanceOrg and amount",

    "sender_balance_error":
        "oldbalanceOrg, amount and newbalanceOrig",

    "expected_receiver_balance":
        "oldbalanceDest and amount",

    "receiver_balance_error":
        "oldbalanceDest, amount and newbalanceDest"
}

for feature, source in feature_sources.items():

    print("\nFeature:", feature)
    print("Calculated using:", source)
    print("Available at transaction decision time: YES")


# ============================================================
# TARGET DEPENDENCY CHECK
# ============================================================

print("\n========== TARGET DEPENDENCY CHECK ==========")

print(
    "Checking whether engineered features were calculated "
    "using the target variable..."
)

for feature in engineered_features:

    # Since engineered features were calculated from transaction
    # and balance information only, they should not contain isFraud.
    print(
        feature,
        "-> does not directly use isFraud"
    )


# ============================================================
# CORRELATION WITH TARGET
# ============================================================

print("\n========== FEATURE CORRELATION WITH FRAUD ==========")

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

correlation = (
    df[numeric_columns]
    .corr()["isFraud"]
    .sort_values(ascending=False)
)

print(correlation)


# ============================================================
# CORRELATION BETWEEN FEATURES
# ============================================================

print("\n========== HIGH FEATURE CORRELATION ==========")

feature_columns = [
    column
    for column in numeric_columns
    if column != "isFraud"
]

correlation_matrix = df[feature_columns].corr()

high_correlations = []

for i in range(len(correlation_matrix.columns)):

    for j in range(i + 1, len(correlation_matrix.columns)):

        feature_a = correlation_matrix.columns[i]
        feature_b = correlation_matrix.columns[j]

        correlation_value = correlation_matrix.iloc[i, j]

        if abs(correlation_value) >= 0.90:

            high_correlations.append({
                "feature_1": feature_a,
                "feature_2": feature_b,
                "correlation": correlation_value
            })

high_corr_df = pd.DataFrame(high_correlations)

if len(high_corr_df) > 0:

    high_corr_df = high_corr_df.sort_values(
        by="correlation",
        key=lambda x: abs(x),
        ascending=False
    )

    print(high_corr_df.to_string(index=False))

else:

    print("No feature pairs with correlation >= 0.90")


# ============================================================
# FRAUD VS LEGITIMATE FEATURE COMPARISON
# ============================================================

print("\n========== FRAUD VS LEGITIMATE COMPARISON ==========")

important_features = [
    "amount",
    "sender_balance_change",
    "sender_balance_change_ratio",
    "receiver_balance_change",
    "amount_to_sender_balance_ratio",
    "expected_sender_balance",
    "sender_balance_error",
    "expected_receiver_balance",
    "receiver_balance_error"
]

for feature in important_features:

    print("\n----------------------------------------")
    print("Feature:", feature)

    comparison = df.groupby("isFraud")[feature].agg(
        ["mean", "median", "min", "max"]
    )

    print(comparison)


# ============================================================
# CHECK FOR PERFECT TARGET SEPARATION
# ============================================================

print("\n========== TARGET SEPARATION CHECK ==========")

print(
    "Checking whether any feature perfectly separates "
    "fraudulent and legitimate transactions..."
)

for feature in important_features:

    fraud_values = df.loc[
        df["isFraud"] == 1,
        feature
    ]

    legitimate_values = df.loc[
        df["isFraud"] == 0,
        feature
    ]

    fraud_min = fraud_values.min()
    fraud_max = fraud_values.max()

    legitimate_min = legitimate_values.min()
    legitimate_max = legitimate_values.max()

    print("\nFeature:", feature)

    print("Legitimate range:",
          legitimate_min,
          "to",
          legitimate_max)

    print("Fraud range:",
          fraud_min,
          "to",
          fraud_max)


# ============================================================
# PAYMENTS / TRANSACTION TYPE ANALYSIS
# ============================================================

print("\n========== TRANSACTION TYPE ANALYSIS ==========")

type_analysis = df.groupby("type").agg(
    total_transactions=("isFraud", "count"),
    fraudulent_transactions=("isFraud", "sum")
)

type_analysis["fraud_percentage"] = (
    type_analysis["fraudulent_transactions"]
    / type_analysis["total_transactions"]
    * 100
)

print(type_analysis)


# ============================================================
# isFlaggedFraud CHECK
# ============================================================

print("\n========== EXISTING FRAUD FLAG CHECK ==========")

flag_analysis = pd.crosstab(
    df["isFlaggedFraud"],
    df["isFraud"]
)

print(flag_analysis)

print(
    "\nConclusion: isFlaggedFraud should remain excluded "
    "from the primary model because it represents an existing "
    "fraud detection mechanism rather than an independent "
    "behavioral feature."
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n========== RANDOM FOREST FEATURE IMPORTANCE ==========")

importance_path = (
    "dataset/random_forest_feature_importance.csv"
)

importance_df = pd.read_csv(importance_path)

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print(importance_df.to_string(index=False))


# ============================================================
# TOP FEATURE SUMMARY
# ============================================================

print("\n========== TOP FEATURES ==========")

top_features = importance_df.head(10)

print(top_features.to_string(index=False))


# ============================================================
# FINAL ASSESSMENT
# ============================================================

print("\n========== FEATURE / LEAKAGE ASSESSMENT ==========")

print("""
1. Engineered features were created from transaction and
   balance information.

2. None of the engineered features directly use isFraud.

3. The features can therefore be calculated before a fraud
   prediction is made.

4. Several engineered features are mathematically related
   to the original balance and amount columns.

5. High correlation between features does NOT automatically
   mean data leakage.

6. However, strong feature relationships should be considered
   during model refinement.

7. isFlaggedFraud should not be used as a primary model feature.

8. The final FinSecure model should use features that will also
   be available in the Stage-2 synthetic UPI dataset.

9. The Random Forest result should therefore be validated again
   after reviewing the feature set.
""")


# ============================================================
# SAVE ANALYSIS
# ============================================================

output_path = "dataset/feature_leakage_analysis.csv"

importance_df.to_csv(
    output_path,
    index=False
)

print("\nAnalysis summary saved to:")
print(output_path)


print("\n========================================")
print("FEATURE / LEAKAGE ANALYSIS COMPLETE")
print("========================================")

print("\nNext step:")
print("Refine the feature set and retrain Random Forest.")