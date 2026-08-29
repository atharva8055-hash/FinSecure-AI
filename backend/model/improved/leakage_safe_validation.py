import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

print("\n========== LEAKAGE-SAFE VALIDATION ==========\n")

# ============================================================
# 1. LOAD DATASET
# ============================================================

print("Loading engineered PaySim dataset...")

df = pd.read_csv("dataset/paysim_engineered.csv")

print("Dataset loaded successfully.")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# 2. SORT BY TRANSACTION TIME
# ============================================================

print("\n========== SORTING BY TRANSACTION TIME ==========")

df = df.sort_values("step").reset_index(drop=True)

print("Dataset sorted by step.")


# ============================================================
# 3. LEAKAGE-SAFE FEATURES
# ============================================================
#
# IMPORTANT:
#
# We intentionally DO NOT use:
#
# newbalanceOrig
# newbalanceDest
# sender_balance_change
# sender_balance_change_ratio
# receiver_balance_change
# amount_to_sender_balance_ratio
# expected_sender_balance
# sender_balance_error
# expected_receiver_balance
# receiver_balance_error
# isFlaggedFraud
#
# These features either depend on post-transaction balances
# or represent an existing fraud detection mechanism.
#
# We use only information available before/at transaction
# decision time.
# ============================================================

base_features = [
    "step",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest"
]

print("\n========== LEAKAGE-SAFE BASE FEATURES ==========")

for feature in base_features:
    print("->", feature)


# ============================================================
# 4. ENCODE TRANSACTION TYPE
# ============================================================

print("\n========== TRANSACTION TYPE ENCODING ==========")

if "type" not in df.columns:
    raise ValueError("Original 'type' column is missing.")

print("Original transaction types:")
print(df["type"].unique())

# One-hot encode transaction type
type_encoded = pd.get_dummies(
    df["type"],
    prefix="type",
    dtype=int
)

# Add encoded columns
df = pd.concat([df, type_encoded], axis=1)

transaction_type_features = [
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER"
]

print("\nEncoded transaction type columns:")

for feature in transaction_type_features:
    print("->", feature)


# ============================================================
# 5. FINAL LEAKAGE-SAFE FEATURE SET
# ============================================================

features = base_features + transaction_type_features

print("\n========== FINAL LEAKAGE-SAFE FEATURE SET ==========")

print(f"Number of leakage-safe features: {len(features)}")

for i, feature in enumerate(features, start=1):
    print(f"{i} -> {feature}")


# ============================================================
# 6. EXCLUDED FEATURES
# ============================================================

excluded_features = [
    "newbalanceOrig",
    "newbalanceDest",
    "sender_balance_change",
    "sender_balance_change_ratio",
    "receiver_balance_change",
    "amount_to_sender_balance_ratio",
    "expected_sender_balance",
    "sender_balance_error",
    "expected_receiver_balance",
    "receiver_balance_error",
    "isFlaggedFraud"
]

print("\n========== EXCLUDED FEATURES ==========")

for feature in excluded_features:
    print("Excluded ->", feature)


# ============================================================
# 7. FEATURE VALIDATION
# ============================================================

print("\n========== FEATURE VALIDATION ==========")

missing_features = [
    feature for feature in features
    if feature not in df.columns
]

if missing_features:
    print("ERROR: Missing features:")

    for feature in missing_features:
        print(feature)

    raise ValueError(
        "Required leakage-safe features are missing."
    )

print("All leakage-safe features are available.")


# ============================================================
# 8. SEPARATE X AND Y
# ============================================================

X = df[features]
y = df["isFraud"]

print("\n========== DATA SHAPE ==========")

print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 9. CLASS DISTRIBUTION
# ============================================================

print("\n========== CLASS DISTRIBUTION ==========")

print(y.value_counts())

print("\nFraud percentage:")

print(
    y.value_counts(normalize=True) * 100
)


# ============================================================
# 10. TIME-AWARE SPLIT
# ============================================================
#
# Train = earlier transaction periods
# Test  = later transaction periods
#
# This prevents future transactions from entering training.
# ============================================================

print("\n========== TIME-AWARE SPLIT ==========")

split_step = int(
    df["step"].quantile(0.80)
)

train_mask = df["step"] <= split_step
test_mask = df["step"] > split_step

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

print("Split step:", split_step)

print(
    "Training time range:",
    df.loc[train_mask, "step"].min(),
    "to",
    df.loc[train_mask, "step"].max()
)

print(
    "Testing time range:",
    df.loc[test_mask, "step"].min(),
    "to",
    df.loc[test_mask, "step"].max()
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# ============================================================
# 11. TRAINING CLASS DISTRIBUTION
# ============================================================

print("\n========== TRAINING CLASS DISTRIBUTION ==========")

print(y_train.value_counts())

print("\nTraining fraud percentage:")

print(
    y_train.value_counts(normalize=True) * 100
)


# ============================================================
# 12. TESTING CLASS DISTRIBUTION
# ============================================================

print("\n========== TESTING CLASS DISTRIBUTION ==========")

print(y_test.value_counts())

print("\nTesting fraud percentage:")

print(
    y_test.value_counts(normalize=True) * 100
)


# ============================================================
# 13. TRAIN RANDOM FOREST
# ============================================================

print("\n========== TRAINING LEAKAGE-SAFE RANDOM FOREST ==========")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("Training model...")

model.fit(X_train, y_train)

print("Model training completed successfully.")


# ============================================================
# 14. PREDICTIONS
# ============================================================

print("\n========== MAKING PREDICTIONS ==========")

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]

print("Predictions generated successfully.")


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(y_test, y_pred)

print(cm)

tn, fp, fn, tp = cm.ravel()

print("\nTrue Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)


# ============================================================
# 16. CLASSIFICATION REPORT
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Fraud"],
        digits=4
    )
)


# ============================================================
# 17. ROC-AUC
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\n========== ROC-AUC ==========")

print("ROC-AUC:", roc_auc)


# ============================================================
# 18. PR-AUC
# ============================================================

pr_auc = average_precision_score(
    y_test,
    y_probability
)

print("\n========== PR-AUC ==========")

print("PR-AUC:", pr_auc)


# ============================================================
# 19. FRAUD DETECTION SUMMARY
# ============================================================

print("\n========== FRAUD DETECTION SUMMARY ==========")

actual_fraud = int(y_test.sum())

detected_fraud = int(tp)

missed_fraud = int(fn)

print("Actual fraudulent transactions:", actual_fraud)

print("Detected fraudulent transactions:", detected_fraud)

print("Missed fraudulent transactions:", missed_fraud)


# ============================================================
# 20. FEATURE IMPORTANCE
# ============================================================

print("\n========== FEATURE IMPORTANCE ==========")

importance_df = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print(importance_df)


# ============================================================
# 21. SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "metric": [
        "ROC-AUC",
        "PR-AUC",
        "True Negatives",
        "False Positives",
        "False Negatives",
        "True Positives",
        "Actual Fraud",
        "Detected Fraud",
        "Missed Fraud"
    ],
    "value": [
        roc_auc,
        pr_auc,
        tn,
        fp,
        fn,
        tp,
        actual_fraud,
        detected_fraud,
        missed_fraud
    ]
})

results.to_csv(
    "dataset/leakage_safe_validation_results.csv",
    index=False
)

importance_df.to_csv(
    "dataset/leakage_safe_feature_importance.csv",
    index=False
)


# ============================================================
# 22. FINAL INTERPRETATION
# ============================================================

print("\n========== LEAKAGE-SAFE VALIDATION COMPLETE ==========")

print("\nModel:")
print("Random Forest")

print("\nFeatures used:", len(features))

print("ROC-AUC:", roc_auc)

print("PR-AUC:", pr_auc)

print("\nResults saved to:")
print("dataset/leakage_safe_validation_results.csv")

print("\nFeature importance saved to:")
print("dataset/leakage_safe_feature_importance.csv")

print("\nNext step:")
print("Compare leakage-safe performance with Random Forest V2.")