import pandas as pd

# ============================================================
# FinSecure AI
# Stage 1 - PaySim Feature Engineering
# ============================================================

# Path to PaySim dataset
file_path = "dataset/PS_20174392719_1491204439457_log.csv"

# Read dataset
print("Loading PaySim dataset...")

df = pd.read_csv(file_path)

print("Dataset loaded successfully.")
print("Original rows:", len(df))
print("Original columns:", len(df.columns))


# ============================================================
# 1. SENDER BALANCE FEATURES
# ============================================================

print("\nCreating sender balance features...")

# Amount removed from sender account
df["sender_balance_change"] = (
    df["oldbalanceOrg"] - df["newbalanceOrig"]
)

# Percentage of sender balance involved
df["sender_balance_change_ratio"] = (
    df["sender_balance_change"]
    / df["oldbalanceOrg"].replace(0, 1)
)


# ============================================================
# 2. RECEIVER BALANCE FEATURES
# ============================================================

print("Creating receiver balance features...")

# Amount added to receiver account
df["receiver_balance_change"] = (
    df["newbalanceDest"] - df["oldbalanceDest"]
)


# ============================================================
# 3. TRANSACTION AMOUNT FEATURES
# ============================================================

print("Creating transaction amount features...")

# Percentage of sender's balance involved in transaction
df["amount_to_sender_balance_ratio"] = (
    df["amount"]
    / df["oldbalanceOrg"].replace(0, 1)
)


# ============================================================
# 4. EXPECTED BALANCE FEATURES
# ============================================================

print("Creating expected balance features...")

# Expected sender balance after transaction
df["expected_sender_balance"] = (
    df["oldbalanceOrg"] - df["amount"]
)

# Difference between expected and actual sender balance
df["sender_balance_error"] = (
    df["newbalanceOrig"] - df["expected_sender_balance"]
)


# Expected receiver balance after transaction
df["expected_receiver_balance"] = (
    df["oldbalanceDest"] + df["amount"]
)

# Difference between expected and actual receiver balance
df["receiver_balance_error"] = (
    df["newbalanceDest"] - df["expected_receiver_balance"]
)


# ============================================================
# 5. DISPLAY CREATED FEATURES
# ============================================================

print("\n========== CREATED FEATURES ==========")

new_features = [
    "sender_balance_change",
    "sender_balance_change_ratio",
    "receiver_balance_change",
    "amount_to_sender_balance_ratio",
    "expected_sender_balance",
    "sender_balance_error",
    "expected_receiver_balance",
    "receiver_balance_error"
]

for feature in new_features:
    print(feature)


# ============================================================
# 6. FEATURE STATISTICS
# ============================================================

print("\n========== FEATURE STATISTICS ==========")

print(df[new_features].describe())


# ============================================================
# 7. FRAUD VS LEGITIMATE ANALYSIS
# ============================================================

print("\n========== FRAUD VS LEGITIMATE FEATURE ANALYSIS ==========")

print("\nSender Balance Change:")

print(
    df.groupby("isFraud")["sender_balance_change"].mean()
)


print("\nSender Balance Change Ratio:")

print(
    df.groupby("isFraud")["sender_balance_change_ratio"].mean()
)


print("\nReceiver Balance Change:")

print(
    df.groupby("isFraud")["receiver_balance_change"].mean()
)


print("\nAmount to Sender Balance Ratio:")

print(
    df.groupby("isFraud")[
        "amount_to_sender_balance_ratio"
    ].mean()
)


# ============================================================
# 8. SAVE ENGINEERED DATASET
# ============================================================

output_path = "dataset/paysim_engineered.csv"

print("\nSaving engineered dataset...")

df.to_csv(
    output_path,
    index=False
)

print("\n========== FEATURE ENGINEERING COMPLETE ==========")

print("Output file:")
print(output_path)

print("Final rows:", len(df))
print("Final columns:", len(df.columns))