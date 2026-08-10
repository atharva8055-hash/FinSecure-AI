import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# FinSecure AI — Train/Test Split
# ============================================================

print("\nLoading prepared ML dataset...")

# Path to prepared ML dataset
file_path = "dataset/paysim_ml_dataset.csv"

# Load dataset
df = pd.read_csv(file_path)

print("Dataset loaded successfully.")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))


# ============================================================
# Separate Features and Target
# ============================================================

print("\n========== SEPARATING FEATURES AND TARGET ==========")

X = df.drop(columns=["isFraud"])
y = df["isFraud"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# Original Class Distribution
# ============================================================

print("\n========== ORIGINAL CLASS DISTRIBUTION ==========")

print(y.value_counts())

print("\nOriginal Fraud Percentage:")
print(y.value_counts(normalize=True) * 100)


# ============================================================
# Train/Test Split
# ============================================================

print("\n========== TRAIN/TEST SPLIT ==========")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)

print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)


# ============================================================
# Training Class Distribution
# ============================================================

print("\n========== TRAINING SET CLASS DISTRIBUTION ==========")

print(y_train.value_counts())

print("\nTraining Fraud Percentage:")
print(y_train.value_counts(normalize=True) * 100)


# ============================================================
# Testing Class Distribution
# ============================================================

print("\n========== TESTING SET CLASS DISTRIBUTION ==========")

print(y_test.value_counts())

print("\nTesting Fraud Percentage:")
print(y_test.value_counts(normalize=True) * 100)


# ============================================================
# Verify Stratification
# ============================================================

print("\n========== STRATIFICATION CHECK ==========")

original_fraud_ratio = y.mean()
train_fraud_ratio = y_train.mean()
test_fraud_ratio = y_test.mean()

print("Original fraud ratio:", original_fraud_ratio)
print("Training fraud ratio:", train_fraud_ratio)
print("Testing fraud ratio:", test_fraud_ratio)


# ============================================================
# Save Training and Testing Datasets
# ============================================================

print("\n========== SAVING DATASETS ==========")

train_df = X_train.copy()
train_df["isFraud"] = y_train

test_df = X_test.copy()
test_df["isFraud"] = y_test


train_output = "dataset/paysim_train.csv"
test_output = "dataset/paysim_test.csv"

train_df.to_csv(train_output, index=False)
test_df.to_csv(test_output, index=False)

print("Training dataset saved:")
print(train_output)

print("Testing dataset saved:")
print(test_output)


# ============================================================
# Final Summary
# ============================================================

print("\n========== TRAIN/TEST SPLIT COMPLETE ==========")

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

print("Training fraud cases:", int(y_train.sum()))
print("Testing fraud cases:", int(y_test.sum()))

print("\nNext step:")
print("Class imbalance handling and baseline model preparation.")