import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

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
# SEPARATE FEATURES AND TARGET
# ============================================================

print("\n========== SEPARATING FEATURES AND TARGET ==========")

target_column = "isFraud"

X_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]

X_test = test_df.drop(columns=[target_column])
y_test = test_df[target_column]

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# CHECK FEATURES
# ============================================================

print("\n========== FEATURE INFORMATION ==========")

print("Features:")
for feature in X_train.columns:
    print("-", feature)


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

print("\n========== TRAINING RANDOM FOREST ==========")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("Model configuration:")
print("Algorithm: Random Forest")
print("Number of trees:", 100)
print("Maximum tree depth:", 12)
print("Minimum samples per leaf:", 2)
print("Class weighting: balanced")
print("Random state: 42")

print("\nTraining model...")

model.fit(X_train, y_train)

print("Random Forest training completed successfully.")


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

print("\nInterpretation:")
print("True Negatives :", tn)
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
# FRAUD DETECTION SUMMARY
# ============================================================

print("\n========== FRAUD DETECTION SUMMARY ==========")

actual_fraud = int(y_test.sum())
detected_fraud = int((y_pred == 1).sum())
missed_fraud = int(((y_test == 1) & (y_pred == 0)).sum())

print("Actual fraudulent transactions:", actual_fraud)
print("Detected fraudulent transactions:", detected_fraud)
print("Missed fraudulent transactions:", missed_fraud)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n========== FEATURE IMPORTANCE ==========")

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print(feature_importance.to_string(index=False))


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

importance_path = "dataset/random_forest_feature_importance.csv"

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
    "model": ["Random Forest"],
    "roc_auc": [roc_auc],
    "pr_auc": [pr_auc],
    "true_negatives": [tn],
    "false_positives": [fp],
    "false_negatives": [fn],
    "true_positives": [tp]
})

results_path = "dataset/random_forest_results.csv"

results.to_csv(
    results_path,
    index=False
)

print("\nModel results saved to:")
print(results_path)


# ============================================================
# COMPLETE
# ============================================================

print("\n========== RANDOM FOREST TRAINING COMPLETE ==========")

print("\nModel:")
print("Random Forest")

print("\nROC-AUC:", roc_auc)
print("PR-AUC:", pr_auc)

print("\nNext step:")
print("Compare Random Forest with Logistic Regression baseline.")