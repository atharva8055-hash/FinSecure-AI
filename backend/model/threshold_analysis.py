import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATASETS
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
# 3. FEATURE SCALING
# ============================================================

print("\n========== FEATURE SCALING ==========")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed.")


# ============================================================
# 4. TRAIN BASELINE LOGISTIC REGRESSION
# ============================================================

print("\n========== TRAINING LOGISTIC REGRESSION ==========")

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("Model training completed successfully.")


# ============================================================
# 5. GENERATE FRAUD PROBABILITIES
# ============================================================

print("\n========== GENERATING FRAUD PROBABILITIES ==========")

y_probability = model.predict_proba(X_test_scaled)[:, 1]

print("Fraud probabilities generated successfully.")

print("\nProbability statistics:")
print("Minimum:", y_probability.min())
print("Maximum:", y_probability.max())
print("Mean:", y_probability.mean())
print("Median:", np.median(y_probability))


# ============================================================
# 6. THRESHOLD ANALYSIS
# ============================================================

print("\n========== THRESHOLD ANALYSIS ==========")

thresholds = [
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

for threshold in thresholds:

    # Convert probability into prediction
    y_pred = (y_probability >= threshold).astype(int)

    # Metrics
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

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()

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


# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n========== THRESHOLD COMPARISON ==========")

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 8. BEST F1 THRESHOLD
# ============================================================

best_f1_row = results_df.loc[
    results_df["f1_score"].idxmax()
]

print("\n========== BEST F1 THRESHOLD ==========")

print(
    "Threshold:",
    best_f1_row["threshold"]
)

print(
    "Precision:",
    best_f1_row["precision"]
)

print(
    "Recall:",
    best_f1_row["recall"]
)

print(
    "F1-score:",
    best_f1_row["f1_score"]
)

print(
    "False Positives:",
    int(best_f1_row["false_positives"])
)

print(
    "False Negatives:",
    int(best_f1_row["false_negatives"])
)


# ============================================================
# 9. SAVE RESULTS
# ============================================================

output_path = "dataset/threshold_analysis.csv"

results_df.to_csv(
    output_path,
    index=False
)

print("\n========== THRESHOLD ANALYSIS COMPLETE ==========")

print("Results saved to:")
print(output_path)

print("\nNext step:")
print("Analyze threshold results and select an appropriate operating threshold.")