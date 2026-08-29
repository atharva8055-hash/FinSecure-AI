import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "dataset/paysim_engineered.csv"

MODEL_DIR = "model/final"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "finsecure_leakage_safe_random_forest.joblib"
)

RESULT_PATH = "dataset/final_model_results.csv"

RANDOM_STATE = 42

N_ESTIMATORS = 100
MAX_DEPTH = 12
MIN_SAMPLES_LEAF = 2

CLASS_WEIGHT = "balanced"

# Final threshold selected during leakage-safe threshold selection
FINAL_THRESHOLD = 0.5


# ============================================================
# LEAKAGE-SAFE BASE FEATURES
# ============================================================

BASE_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest"
]

TARGET = "isFraud"


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


# ============================================================
# LOAD DATASET
# ============================================================

print("\n========== FINAL FINSECURE MODEL ==========")

print("\n========== LOADING DATASET ==========")

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

TYPE_COLUMNS = [
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER"
]


if "type" in df.columns:

    print("Original transaction types:")
    print(df["type"].unique())

    type_encoded = pd.get_dummies(
        df["type"],
        prefix="type",
        dtype=int
    )

    for column in TYPE_COLUMNS:

        if column not in type_encoded.columns:
            type_encoded[column] = 0

    type_encoded = type_encoded[TYPE_COLUMNS]

    df = pd.concat(
        [
            df.drop(columns=["type"]),
            type_encoded
        ],
        axis=1
    )

    print("\nEncoded transaction type columns:")

    for column in TYPE_COLUMNS:
        print(f"-> {column}")


else:

    print("Transaction type already encoded.")


# ============================================================
# FINAL FEATURE SET
# ============================================================

FEATURES = BASE_FEATURES + TYPE_COLUMNS

print("\n========== FINAL LEAKAGE-SAFE FEATURE SET ==========")

print(f"Number of features: {len(FEATURES)}")

for index, feature in enumerate(FEATURES, start=1):
    print(f"{index} -> {feature}")


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
        print(f"-> {feature}")

    raise ValueError(
        "Required leakage-safe features are missing."
    )


if TARGET not in df.columns:

    raise ValueError(
        "Target variable 'isFraud' is missing."
    )


print("All final leakage-safe features are available.")


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

print("\n========== PREPARING DATA ==========")

X = df[FEATURES].copy()
y = df[TARGET].copy()

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n========== CLASS DISTRIBUTION ==========")

class_distribution = y.value_counts()

print(class_distribution)

fraud_percentage = y.value_counts(
    normalize=True
) * 100

print("\nFraud percentage:")

print(fraud_percentage)


# ============================================================
# TIME-AWARE TRAINING
# ============================================================

print("\n========== TIME-AWARE TRAINING ==========")

# Same time boundary used during leakage-safe validation
SPLIT_STEP = 355

train_mask = df["step"] <= SPLIT_STEP
test_mask = df["step"] > SPLIT_STEP

X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

X_test = X.loc[test_mask]
y_test = y.loc[test_mask]

print(f"Training time range: 1 to {SPLIT_STEP}")
print(
    f"Testing time range: {SPLIT_STEP + 1} to "
    f"{df['step'].max()}"
)

print(f"\nTraining rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")


# ============================================================
# TRAINING CLASS DISTRIBUTION
# ============================================================

print("\n========== TRAINING CLASS DISTRIBUTION ==========")

print(y_train.value_counts())

print("\n========== TESTING CLASS DISTRIBUTION ==========")

print(y_test.value_counts())


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print("\n========== TRAINING FINAL RANDOM FOREST ==========")

print("Model configuration:")
print("Algorithm: Random Forest")
print(f"Number of trees: {N_ESTIMATORS}")
print(f"Maximum tree depth: {MAX_DEPTH}")
print(f"Minimum samples per leaf: {MIN_SAMPLES_LEAF}")
print(f"Class weighting: {CLASS_WEIGHT}")
print(f"Random state: {RANDOM_STATE}")


model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    class_weight=CLASS_WEIGHT,
    random_state=RANDOM_STATE,
    n_jobs=-1
)


print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Final Random Forest training completed successfully.")


# ============================================================
# GENERATE FRAUD PROBABILITIES
# ============================================================

print("\n========== GENERATING FRAUD PROBABILITIES ==========")

fraud_probabilities = model.predict_proba(
    X_test
)[:, 1]

print("Fraud probabilities generated successfully.")


# ============================================================
# APPLY FINAL THRESHOLD
# ============================================================

print("\n========== APPLYING FINAL THRESHOLD ==========")

print(
    f"Final operating threshold: "
    f"{FINAL_THRESHOLD}"
)

y_pred = (
    fraud_probabilities >= FINAL_THRESHOLD
).astype(int)

print("Final fraud decisions generated.")


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n========== CONFUSION MATRIX ==========")

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred
).ravel()

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print(f"\nTrue Negatives : {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives : {tp}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Legitimate",
            "Fraud"
        ],
        digits=4
    )
)


# ============================================================
# MODEL METRICS
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    fraud_probabilities
)

pr_auc = average_precision_score(
    y_test,
    fraud_probabilities
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n========== FINAL MODEL METRICS ==========")

print(f"ROC-AUC : {roc_auc:.6f}")
print(f"PR-AUC  : {pr_auc:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall   : {recall:.6f}")
print(f"F1-score : {f1:.6f}")


# ============================================================
# FRAUD DETECTION SUMMARY
# ============================================================

actual_fraud = int(y_test.sum())

detected_fraud = int(tp)

missed_fraud = int(fn)


print("\n========== FRAUD DETECTION SUMMARY ==========")

print(
    f"Actual fraudulent transactions: "
    f"{actual_fraud}"
)

print(
    f"Detected fraudulent transactions: "
    f"{detected_fraud}"
)

print(
    f"Missed fraudulent transactions: "
    f"{missed_fraud}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n========== FEATURE IMPORTANCE ==========")

feature_importance = pd.DataFrame(
    {
        "feature": FEATURES,
        "importance": model.feature_importances_
    }
).sort_values(
    "importance",
    ascending=False
)

print(feature_importance)


# ============================================================
# SAVE MODEL
# ============================================================

print("\n========== SAVING FINAL MODEL ==========")

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)

print(
    f"Trained model saved to:\n"
    f"{MODEL_PATH}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame(
    [
        ["model", "Leakage-Safe Random Forest"],
        ["validation", "Time-Aware"],
        ["features", len(FEATURES)],
        ["split_step", SPLIT_STEP],
        ["threshold", FINAL_THRESHOLD],
        ["n_estimators", N_ESTIMATORS],
        ["max_depth", MAX_DEPTH],
        ["min_samples_leaf", MIN_SAMPLES_LEAF],
        ["class_weight", CLASS_WEIGHT],
        ["roc_auc", roc_auc],
        ["pr_auc", pr_auc],
        ["precision", precision],
        ["recall", recall],
        ["f1_score", f1],
        ["true_negatives", tn],
        ["false_positives", fp],
        ["false_negatives", fn],
        ["true_positives", tp]
    ],
    columns=[
        "metric",
        "value"
    ]
)

results.to_csv(
    RESULT_PATH,
    index=False
)

print(
    f"Results saved to:\n"
    f"{RESULT_PATH}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("FINAL FINSECURE MODEL COMPLETE")
print("========================================")

print("\nModel:")
print("Leakage-Safe Random Forest")

print("\nValidation:")
print("Time-Aware Validation")

print("\nFeatures:")
print(f"{len(FEATURES)} leakage-safe features")

print("\nOperating Threshold:")
print(FINAL_THRESHOLD)

print("\nROC-AUC:")
print(f"{roc_auc:.6f}")

print("\nPR-AUC:")
print(f"{pr_auc:.6f}")

print("\nPrecision:")
print(f"{precision:.6f}")

print("\nRecall:")
print(f"{recall:.6f}")

print("\nF1-score:")
print(f"{f1:.6f}")

print("\nFalse Positives:")
print(fp)

print("\nFalse Negatives:")
print(fn)

print("\nTrue Positives:")
print(tp)

print("\nFinal model file:")
print(MODEL_PATH)

print("\nNext step:")
print(
    "Create the fraud prediction inference module "
    "for FinSecure AI."
)