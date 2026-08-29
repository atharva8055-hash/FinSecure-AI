import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "dataset/paysim_engineered.csv"
OUTPUT_PATH = "dataset/leakage_safe_threshold_analysis.csv"

TARGET = "isFraud"

RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = 12
MIN_SAMPLES_LEAF = 2
CLASS_WEIGHT = "balanced"

# IMPORTANT:
# This is the exact split used in leakage_safe_validation.py
SPLIT_STEP = 355


# ============================================================
# LEAKAGE-SAFE BASE FEATURES
# ============================================================

BASE_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest"
]


# ============================================================
# TRANSACTION TYPES
# ============================================================

TRANSACTION_TYPES = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER"
]


# ============================================================
# HEADER
# ============================================================

print("\n========== LEAKAGE-SAFE THRESHOLD ANALYSIS ==========")


# ============================================================
# LOAD DATASET
# ============================================================

print("\n========== LOADING ENGINEERED DATASET ==========")

print("Loading engineered PaySim dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# SORT BY TRANSACTION TIME
# ============================================================

print("\n========== SORTING BY TRANSACTION TIME ==========")

df = df.sort_values("step").reset_index(drop=True)

print("Dataset sorted by step.")


# ============================================================
# TRANSACTION TYPE ENCODING
# ============================================================

print("\n========== TRANSACTION TYPE ENCODING ==========")

if "type" not in df.columns:
    raise ValueError(
        "Transaction type column 'type' is missing."
    )

print("Original transaction types:")
print(df["type"].unique())

df = pd.get_dummies(
    df,
    columns=["type"],
    prefix="type",
    dtype=int
)

print("\nEncoded transaction type columns:")

for transaction_type in TRANSACTION_TYPES:

    column = f"type_{transaction_type}"

    if column not in df.columns:
        df[column] = 0

    print(f"-> {column}")


# ============================================================
# FINAL LEAKAGE-SAFE FEATURE SET
# ============================================================

FEATURES = BASE_FEATURES + [
    f"type_{transaction_type}"
    for transaction_type in TRANSACTION_TYPES
]


print("\n========== FINAL LEAKAGE-SAFE FEATURE SET ==========")

print(
    f"Number of leakage-safe features: {len(FEATURES)}"
)

for index, feature in enumerate(FEATURES, start=1):
    print(f"{index} -> {feature}")


# ============================================================
# EXCLUDED FEATURES
# ============================================================

EXCLUDED_FEATURES = [
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

for feature in EXCLUDED_FEATURES:
    print(f"Excluded -> {feature}")


# ============================================================
# FEATURE VALIDATION
# ============================================================

print("\n========== FEATURE VALIDATION ==========")

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:

    print("ERROR: Missing features:")

    for feature in missing_features:
        print(feature)

    raise ValueError(
        "Required leakage-safe features are missing."
    )


if TARGET not in df.columns:

    raise ValueError(
        "Target variable isFraud is missing."
    )


print("All leakage-safe features are available.")


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

print("\n========== DATA SHAPE ==========")

X = df[FEATURES]
y = df[TARGET]

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n========== CLASS DISTRIBUTION ==========")

print(y.value_counts())

print("\nFraud percentage:")

print(
    y.value_counts(normalize=True) * 100
)


# ============================================================
# EXACT TIME-AWARE SPLIT
# ============================================================

print("\n========== TIME-AWARE SPLIT ==========")

print(
    f"Using fixed split step: {SPLIT_STEP}"
)

train_mask = df["step"] <= SPLIT_STEP
test_mask = df["step"] > SPLIT_STEP

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]


print(
    f"Training time range: "
    f"{df.loc[train_mask, 'step'].min()} "
    f"to "
    f"{df.loc[train_mask, 'step'].max()}"
)

print(
    f"Testing time range: "
    f"{df.loc[test_mask, 'step'].min()} "
    f"to "
    f"{df.loc[test_mask, 'step'].max()}"
)

print(f"\nTraining rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")


# ============================================================
# TRAINING CLASS DISTRIBUTION
# ============================================================

print("\n========== TRAINING CLASS DISTRIBUTION ==========")

print(y_train.value_counts())

print("\nTraining fraud percentage:")

print(
    y_train.value_counts(normalize=True) * 100
)


# ============================================================
# TESTING CLASS DISTRIBUTION
# ============================================================

print("\n========== TESTING CLASS DISTRIBUTION ==========")

print(y_test.value_counts())

print("\nTesting fraud percentage:")

print(
    y_test.value_counts(normalize=True) * 100
)


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print("\n========== TRAINING LEAKAGE-SAFE RANDOM FOREST ==========")

print("Model configuration:")
print("Algorithm: Random Forest")
print(f"Number of trees: {N_ESTIMATORS}")
print(f"Maximum tree depth: {MAX_DEPTH}")
print(f"Minimum samples per leaf: {MIN_SAMPLES_LEAF}")
print(f"Class weighting: {CLASS_WEIGHT}")
print(f"Random state: {RANDOM_STATE}")

print("\nTraining model...")

model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    class_weight=CLASS_WEIGHT,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print(
    "Leakage-safe Random Forest training "
    "completed successfully."
)


# ============================================================
# GENERATE FRAUD PROBABILITIES
# ============================================================

print("\n========== GENERATING FRAUD PROBABILITIES ==========")

fraud_probabilities = model.predict_proba(
    X_test
)[:, 1]

print(
    "Fraud probabilities generated successfully."
)


# ============================================================
# PROBABILITY STATISTICS
# ============================================================

print("\nProbability statistics:")

print(
    f"Minimum: {fraud_probabilities.min()}"
)

print(
    f"Maximum: {fraud_probabilities.max()}"
)

print(
    f"Mean: {fraud_probabilities.mean()}"
)

print(
    f"Median: "
    f"{pd.Series(fraud_probabilities).median()}"
)


# ============================================================
# ROC-AUC
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    fraud_probabilities
)


# ============================================================
# PR-AUC
# ============================================================

pr_auc = average_precision_score(
    y_test,
    fraud_probabilities
)


print("\n========== MODEL PROBABILITY METRICS ==========")

print(
    f"ROC-AUC: {roc_auc}"
)

print(
    f"PR-AUC: {pr_auc}"
)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n========== THRESHOLD ANALYSIS ==========")

# These thresholds will be compared against
# exactly the same time-aware test set.

THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]

results = []


for threshold in THRESHOLDS:

    predictions = (
        fraud_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    results.append({

        "threshold": threshold,

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "true_negatives": tn,

        "false_positives": fp,

        "false_negatives": fn,

        "true_positives": tp

    })


results_df = pd.DataFrame(results)


# ============================================================
# DISPLAY THRESHOLD RESULTS
# ============================================================

print("\n========== THRESHOLD COMPARISON ==========")

print(
    results_df.to_string(index=False)
)


# ============================================================
# BEST F1 THRESHOLD
# ============================================================

best_f1_row = results_df.loc[
    results_df["f1_score"].idxmax()
]


print("\n========== BEST F1 THRESHOLD ==========")

print(
    f"Threshold: "
    f"{best_f1_row['threshold']}"
)

print(
    f"Precision: "
    f"{best_f1_row['precision']}"
)

print(
    f"Recall: "
    f"{best_f1_row['recall']}"
)

print(
    f"F1-score: "
    f"{best_f1_row['f1_score']}"
)

print(
    f"False Positives: "
    f"{int(best_f1_row['false_positives'])}"
)

print(
    f"False Negatives: "
    f"{int(best_f1_row['false_negatives'])}"
)

print(
    f"True Positives: "
    f"{int(best_f1_row['true_positives'])}"
)


# ============================================================
# HIGH-RECALL THRESHOLD
# ============================================================

print("\n========== HIGH-RECALL THRESHOLD ==========")

# We want the highest threshold that still
# maintains at least 90% recall.

high_recall_candidates = results_df[
    results_df["recall"] >= 0.90
]

if not high_recall_candidates.empty:

    high_recall_row = (
        high_recall_candidates
        .iloc[-1]
    )

    print(
        f"Threshold: "
        f"{high_recall_row['threshold']}"
    )

    print(
        f"Precision: "
        f"{high_recall_row['precision']}"
    )

    print(
        f"Recall: "
        f"{high_recall_row['recall']}"
    )

    print(
        f"F1-score: "
        f"{high_recall_row['f1_score']}"
    )

    print(
        f"False Positives: "
        f"{int(high_recall_row['false_positives'])}"
    )

    print(
        f"False Negatives: "
        f"{int(high_recall_row['false_negatives'])}"
    )

else:

    print(
        "No tested threshold achieved "
        "at least 90% recall."
    )


# ============================================================
# DEFAULT THRESHOLD
# ============================================================

default_row = results_df[
    results_df["threshold"] == 0.50
].iloc[0]


# ============================================================
# DEFAULT VS BEST F1
# ============================================================

print("\n========== DEFAULT VS SELECTED ==========")

print("\nDefault Threshold (0.5)")

print(
    f"Precision: "
    f"{default_row['precision']}"
)

print(
    f"Recall: "
    f"{default_row['recall']}"
)

print(
    f"F1-score: "
    f"{default_row['f1_score']}"
)

print(
    f"False Positives: "
    f"{int(default_row['false_positives'])}"
)

print(
    f"False Negatives: "
    f"{int(default_row['false_negatives'])}"
)


print("\nBest F1 Threshold")

print(
    f"Threshold: "
    f"{best_f1_row['threshold']}"
)

print(
    f"Precision: "
    f"{best_f1_row['precision']}"
)

print(
    f"Recall: "
    f"{best_f1_row['recall']}"
)

print(
    f"F1-score: "
    f"{best_f1_row['f1_score']}"
)

print(
    f"False Positives: "
    f"{int(best_f1_row['false_positives'])}"
)

print(
    f"False Negatives: "
    f"{int(best_f1_row['false_negatives'])}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

print("\n========== SAVING RESULTS ==========")

os.makedirs(
    "dataset",
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    "Threshold analysis results saved to:"
)

print(
    OUTPUT_PATH
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n========== LEAKAGE-SAFE THRESHOLD "
    "ANALYSIS COMPLETE =========="
)

print("\nModel:")
print("Leakage-Safe Random Forest")

print(
    f"Features used: {len(FEATURES)}"
)

print(
    f"Time split: step <= {SPLIT_STEP} "
    "for training"
)

print(
    f"ROC-AUC: {roc_auc}"
)

print(
    f"PR-AUC: {pr_auc}"
)

print(
    "\nRecommended threshold based on "
    "best F1-score:"
)

print(
    f"{best_f1_row['threshold']}"
)

print(
    "\nNext step:"
)

print(
    "Analyze the threshold results and "
    "select the final operating threshold "
    "for FinSecure AI."
)