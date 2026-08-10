import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# LOAD DATASETS
# ============================================================

print("\n========== LOADING DATASETS ==========")

train_path = "dataset/paysim_train.csv"
test_path = "dataset/paysim_test.csv"

print("Loading training dataset...")
train_df = pd.read_csv(train_path)

print("Loading testing dataset...")
test_df = pd.read_csv(test_path)

print("Datasets loaded successfully.")

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))


# ============================================================
# TARGET
# ============================================================

target = "isFraud"


# ============================================================
# REFINED FEATURE LIST
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


# ============================================================
# FEATURE INFORMATION
# ============================================================

print("\n========== REFINED FEATURE SET ==========")

print("Number of features:", len(features))

for index, feature in enumerate(features, start=1):
    print(index, "->", feature)


print("\nRemoved redundant features:")

print(" - expected_sender_balance")
print(" - expected_receiver_balance")


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

print("\n========== SEPARATING FEATURES AND TARGET ==========")

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# FEATURE VALIDATION
# ============================================================

print("\n========== FEATURE VALIDATION ==========")

missing_train = [
    feature for feature in features
    if feature not in train_df.columns
]

missing_test = [
    feature for feature in features
    if feature not in test_df.columns
]

if missing_train:
    print("Missing features in training dataset:")
    print(missing_train)
    raise ValueError("Required training features are missing.")

if missing_test:
    print("Missing features in testing dataset:")
    print(missing_test)
    raise ValueError("Required testing features are missing.")

print("All refined features are available.")


# ============================================================
# CHECK CLASS DISTRIBUTION
# ============================================================

print("\n========== CLASS DISTRIBUTION ==========")

print("Training target:")
print(y_train.value_counts())

print("\nTesting target:")
print(y_test.value_counts())


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print("\n========== TRAINING RANDOM FOREST V2 ==========")

print("Model configuration:")
print("Algorithm: Random Forest")
print("Number of trees: 100")
print("Maximum tree depth: 12")
print("Minimum samples per leaf: 2")
print("Class weighting: balanced")
print("Random state: 42")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")

model.fit(X_train, y_train)

print("Random Forest V2 training completed successfully.")


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

true_negatives = cm[0][0]
false_positives = cm[0][1]
false_negatives = cm[1][0]
true_positives = cm[1][1]

print("\nInterpretation:")

print("True Negatives :", true_negatives)
print("False Positives:", false_positives)
print("False Negatives:", false_negatives)
print("True Positives :", true_positives)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========")

report = classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate", "Fraud"],
    digits=4
)

print(report)


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
# FRAUD DETECTION SUMMARY
# ============================================================

print("\n========== FRAUD DETECTION SUMMARY ==========")

actual_fraud = int(y_test.sum())

predicted_fraud = int(y_pred.sum())

missed_fraud = int(
    ((y_test == 1) & (y_pred == 0)).sum()
)

print("Actual fraudulent transactions:", actual_fraud)

print("Predicted fraudulent transactions:", predicted_fraud)

print("Missed fraudulent transactions:", missed_fraud)

print("Detected fraudulent transactions:", true_positives)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n========== FEATURE IMPORTANCE ==========")

feature_importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print(
    feature_importance.to_string(index=False)
)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

importance_path = (
    "dataset/random_forest_v2_feature_importance.csv"
)

feature_importance.to_csv(
    importance_path,
    index=False
)

print("\nFeature importance saved to:")
print(importance_path)


# ============================================================
# SAVE MODEL RESULTS
# ============================================================

results = pd.DataFrame({
    "model": ["Random Forest V2"],
    "n_estimators": [100],
    "max_depth": [12],
    "min_samples_leaf": [2],
    "class_weight": ["balanced"],
    "roc_auc": [roc_auc],
    "pr_auc": [pr_auc],
    "true_negatives": [true_negatives],
    "false_positives": [false_positives],
    "false_negatives": [false_negatives],
    "true_positives": [true_positives]
})

results_path = "dataset/random_forest_v2_results.csv"

results.to_csv(
    results_path,
    index=False
)

print("\nModel results saved to:")
print(results_path)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n========================================")
print("RANDOM FOREST V2 TRAINING COMPLETE")
print("========================================")

print("\nModel: Random Forest V2")

print("Features:", len(features))

print("ROC-AUC:", roc_auc)

print("PR-AUC:", pr_auc)

print("\nNext step:")
print("Compare Random Forest V1 and Random Forest V2.")