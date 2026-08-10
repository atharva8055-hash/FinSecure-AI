import pandas as pd

# ============================================================
# LOAD DATASET
# ============================================================

# Path to the PaySim dataset
file_path = "dataset/PS_20174392719_1491204439457_log.csv"

# Read the CSV file
df = pd.read_csv(file_path)


# ============================================================
# BASIC DATASET INFORMATION
# ============================================================

print("\n========== DATASET SHAPE ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n========== COLUMN NAMES ==========")
for column in df.columns:
    print(column)

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


# ============================================================
# FRAUD DISTRIBUTION
# ============================================================

print("\n========== FRAUD DISTRIBUTION ==========")
print(df["isFraud"].value_counts())

print("\n========== FRAUD PERCENTAGE ==========")
print(df["isFraud"].value_counts(normalize=True) * 100)


# ============================================================
# 1. TRANSACTION TYPE ANALYSIS
# ============================================================

print("\n\n========== TRANSACTION TYPE ANALYSIS ==========")

# Number of transactions for each transaction type
print("\nTransaction Count by Type:")
print(df["type"].value_counts())

# Number of fraudulent transactions for each transaction type
print("\nFraud Count by Transaction Type:")
print(
    df[df["isFraud"] == 1]["type"].value_counts()
)

# Calculate fraud percentage for each transaction type
print("\nFraud Percentage by Transaction Type:")

fraud_by_type = (
    df.groupby("type")["isFraud"]
    .agg(["count", "sum"])
)

# Rename columns for better readability
fraud_by_type.columns = [
    "total_transactions",
    "fraud_transactions"
]

# Calculate percentage
fraud_by_type["fraud_percentage"] = (
    fraud_by_type["fraud_transactions"]
    / fraud_by_type["total_transactions"]
    * 100
)

print(fraud_by_type)


# ============================================================
# 2. TRANSACTION AMOUNT ANALYSIS
# ============================================================

print("\n\n========== TRANSACTION AMOUNT ANALYSIS ==========")

# Overall transaction amount statistics
print("\nOverall Transaction Amount Statistics:")
print(df["amount"].describe())

# Legitimate transaction statistics
print("\nLegitimate Transaction Amount Statistics:")
print(
    df[df["isFraud"] == 0]["amount"].describe()
)

# Fraudulent transaction statistics
print("\nFraudulent Transaction Amount Statistics:")
print(
    df[df["isFraud"] == 1]["amount"].describe()
)


# ============================================================
# 3. BALANCE BEHAVIOR ANALYSIS
# ============================================================

print("\n\n========== BALANCE BEHAVIOR ANALYSIS ==========")

# Select only fraudulent transactions
fraud_df = df[df["isFraud"] == 1]

# Sender balance analysis
print("\nFraudulent Transactions - Sender Balance:")
print(
    fraud_df[
        ["oldbalanceOrg", "newbalanceOrig"]
    ].describe()
)

# Receiver balance analysis
print("\nFraudulent Transactions - Receiver Balance:")
print(
    fraud_df[
        ["oldbalanceDest", "newbalanceDest"]
    ].describe()
)


# ============================================================
# 4. EXISTING FRAUD FLAG ANALYSIS
# ============================================================

print("\n\n========== EXISTING FRAUD FLAG ANALYSIS ==========")

# Distribution of isFlaggedFraud
print("\nisFlaggedFraud Distribution:")
print(df["isFlaggedFraud"].value_counts())

# Number of actual fraud transactions that were flagged
flagged_fraud_count = df[
    (df["isFraud"] == 1) &
    (df["isFlaggedFraud"] == 1)
].shape[0]

print(
    "\nFraudulent transactions with isFlaggedFraud = 1:",
    flagged_fraud_count
)


# ============================================================
# ANALYSIS COMPLETE
# ============================================================

print("\n\n========== DATASET INSPECTION COMPLETE ==========")