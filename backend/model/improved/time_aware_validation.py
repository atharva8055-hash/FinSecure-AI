import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


print("\n========== TIME-AWARE VALIDATION ==========")


# ============================================================
# LOAD ENGINEERED DATASET
# ============================================================

print("\nLoading engineered dataset...")

file_path = "dataset/paysim_ml_dataset.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# SORT BY TRANSACTION TIME
# ============================================================

print("\n========== SORTING BY TRANSACTION TIME ==========")

df = df.sort_values("step").reset_index(drop=True)

print("Dataset sorted by step.")


# ============================================================
# DEFINE FEATURES
# ============================================================

features = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "sender_balance_change",
    "sender_balance_change_ratio",
    "receiver_balance_change",
    "amount_to_sender_balance_ratio",
    "sender_balance_error",
    "receiver_balance_error",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER"
]

target = "isFraud"


print("\n========== FEATURE SET ==========")

print("Number of features:", len(features))

for i, feature in enumerate(features, start=1):
    print(f"{i} -> {feature}")


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X = df[features]
y = df[target]


print("\n========== DATA SHAPE ==========")

print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# TIME-AWARE SPLIT
# ============================================================

print("\n========== TIME-AWARE TRAIN/TEST SPLIT ==========")

unique_steps = sorted(df["step"].unique())

split_index = int(len(unique_steps) * 0.80)

train_steps = unique_steps[:split_index]
test_steps = unique_steps[split_index:]


train_df = df[df["step"].isin(train_steps)]
test_df = df[df["step"].isin(test_steps)]


X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]


print("Training time range:")
print(
    train_df["step"].min(),
    "to",
    train_df["step"].max()
)

print("Testing time range:")
print(
    test_df["step"].min(),
    "to",
    test_df["step"].max()
)


print("\nTraining rows:", len(train_df))
print("Testing rows:", len(test_df))


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n========== TRAINING CLASS DISTRIBUTION ==========")

print(y_train.value_counts())

print("\nTraining fraud percentage:")
print(y_train.value_counts(normalize=True) * 100)


print("\n========== TESTING CLASS DISTRIBUTION ==========")

print(y_test.value_counts())

print("\nTesting fraud percentage:")
print(y_test.value_counts(normalize=True) * 100)


# ============================================================
# TRAIN RANDOM FOREST V2
# ============================================================

print("\n========== TRAINING RANDOM FOREST V2 ==========")

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
# PREDICTIONS
# ============================================================

print("\n========== MAKING PREDICTIONS ==========")

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]

print("Predictions generated successfully.")


# ============================================================
# CONFUSION MATRIX
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
# CLASSIFICATION REPORT
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
# ROC-AUC
# ============================================================

print("\n========== ROC-AUC ==========")

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("ROC-AUC:", roc_auc)


# ============================================================
# PR-AUC
# ============================================================

print("\n========== PR-AUC ==========")

pr_auc = average_precision_score(
    y_test,
    y_probability
)

print("PR-AUC:", pr_auc)


# ============================================================
# FRAUD SUMMARY
# ============================================================

print("\n========== FRAUD DETECTION SUMMARY ==========")

actual_fraud = int(y_test.sum())
detected_fraud = int(tp)
missed_fraud = int(fn)

print("Actual fraudulent transactions:", actual_fraud)
print("Detected fraudulent transactions:", detected_fraud)
print("Missed fraudulent transactions:", missed_fraud)


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "model": ["Random Forest V2"],
    "validation_type": ["Time-Aware"],
    "train_start_step": [train_df["step"].min()],
    "train_end_step": [train_df["step"].max()],
    "test_start_step": [test_df["step"].min()],
    "test_end_step": [test_df["step"].max()],
    "training_rows": [len(train_df)],
    "testing_rows": [len(test_df)],
    "roc_auc": [roc_auc],
    "pr_auc": [pr_auc],
    "true_negatives": [tn],
    "false_positives": [fp],
    "false_negatives": [fn],
    "true_positives": [tp]
})


output_path = "dataset/time_aware_validation_results.csv"

results.to_csv(
    output_path,
    index=False
)


print("\nResults saved to:")
print(output_path)


print("\n========================================")
print("TIME-AWARE VALIDATION COMPLETE")
print("========================================")

print("\nNext step:")
print("Compare random-split and time-aware performance.")