import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


# ============================================================
# FinSecure AI — Class Imbalance Analysis
# ============================================================

print("\nLoading training dataset...")

file_path = "dataset/paysim_train.csv"

df = pd.read_csv(file_path)

print("Training dataset loaded successfully.")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n========== CLASS DISTRIBUTION ==========")

class_distribution = df["isFraud"].value_counts()

print(class_distribution)


# ============================================================
# CLASS PERCENTAGE
# ============================================================

print("\n========== CLASS PERCENTAGE ==========")

class_percentage = df["isFraud"].value_counts(normalize=True) * 100

print(class_percentage)


# ============================================================
# LEGITIMATE VS FRAUD RATIO
# ============================================================

legitimate_count = class_distribution.get(0, 0)
fraud_count = class_distribution.get(1, 0)

print("\n========== CLASS IMBALANCE RATIO ==========")

print("Legitimate transactions:", legitimate_count)
print("Fraudulent transactions:", fraud_count)

if fraud_count > 0:
    ratio = legitimate_count / fraud_count

    print(
        f"Legitimate : Fraud ratio = "
        f"{ratio:.2f} : 1"
    )


# ============================================================
# COMPUTE CLASS WEIGHTS
# ============================================================

print("\n========== CLASS WEIGHTS ==========")

classes = np.array([0, 1])

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=df["isFraud"]
)

class_weights = dict(zip(classes, weights))

print("Class weights:")

for class_label, weight in class_weights.items():

    if class_label == 0:
        label = "Legitimate"

    else:
        label = "Fraud"

    print(
        f"{label} ({class_label}): "
        f"{weight:.4f}"
    )


# ============================================================
# INTERPRETATION
# ============================================================

print("\n========== INTERPRETATION ==========")

print(
    "The dataset is highly imbalanced because "
    "fraudulent transactions represent only a very small "
    "percentage of all transactions."
)

print(
    "Class weighting will be used during model training "
    "so that fraudulent transactions receive greater "
    "importance than legitimate transactions."
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========== CLASS IMBALANCE ANALYSIS COMPLETE ==========")

print("Training rows:", len(df))
print("Legitimate transactions:", legitimate_count)
print("Fraudulent transactions:", fraud_count)

print("\nRecommended approach:")
print("1. Use class-weighted baseline models")
print("2. Evaluate Precision, Recall and F1-score")
print("3. Evaluate PR-AUC / ROC-AUC")
print("4. Compare with other imbalance techniques later")

print("\nNext step:")
print("Baseline model training.")