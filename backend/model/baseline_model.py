import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. LOAD TRAINING AND TESTING DATA
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
# 2. SEPARATE FEATURES AND TARGET
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
# 3. CHECK FEATURE TYPES
# ============================================================

print("\n========== FEATURE TYPES ==========")

print(X_train.dtypes)


# ============================================================
# 4. SCALE NUMERICAL FEATURES
# ============================================================

print("\n========== FEATURE SCALING ==========")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed.")


# ============================================================
# 5. TRAIN LOGISTIC REGRESSION
# ============================================================

print("\n========== TRAINING BASELINE MODEL ==========")

print("Model: Logistic Regression")
print("Class weighting: balanced")

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    solver="lbfgs"
)

model.fit(X_train_scaled, y_train)

print("Model training completed successfully.")


# ============================================================
# 6. MAKE PREDICTIONS
# ============================================================

print("\n========== MAKING PREDICTIONS ==========")

y_pred = model.predict(X_test_scaled)

# Probability of transaction being fraudulent
y_probability = model.predict_proba(X_test_scaled)[:, 1]

print("Predictions generated successfully.")


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(y_test, y_pred)

print(cm)

print("\nInterpretation:")

print("True Negatives :", cm[0][0])
print("False Positives:", cm[0][1])
print("False Negatives:", cm[1][0])
print("True Positives :", cm[1][1])


# ============================================================
# 8. CLASSIFICATION REPORT
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
# 9. ROC-AUC
# ============================================================

print("\n========== ROC-AUC ==========")

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("ROC-AUC:", roc_auc)


# ============================================================
# 10. PR-AUC
# ============================================================

print("\n========== PR-AUC ==========")

pr_auc = average_precision_score(
    y_test,
    y_probability
)

print("PR-AUC:", pr_auc)


# ============================================================
# 11. FRAUD DETECTION SUMMARY
# ============================================================

print("\n========== FRAUD DETECTION SUMMARY ==========")

actual_fraud = (y_test == 1).sum()
detected_fraud = ((y_test == 1) & (y_pred == 1)).sum()
missed_fraud = ((y_test == 1) & (y_pred == 0)).sum()

print("Actual fraudulent transactions:", actual_fraud)
print("Detected fraudulent transactions:", detected_fraud)
print("Missed fraudulent transactions:", missed_fraud)


# ============================================================
# 12. BASELINE MODEL COMPLETE
# ============================================================

print("\n========================================")
print("BASELINE MODEL TRAINING COMPLETE")
print("========================================")

print("\nModel: Logistic Regression")
print("Class Weighting: Balanced")
print("ROC-AUC:", roc_auc)
print("PR-AUC:", pr_auc)

print("\nNext step:")
print("Baseline model evaluation and interpretation.")