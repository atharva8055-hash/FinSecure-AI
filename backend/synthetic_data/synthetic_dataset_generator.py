import pandas as pd
import random
import uuid
from datetime import datetime, timedelta


# ============================================================
# CONFIGURATION
# ============================================================

NUMBER_OF_TRANSACTIONS = 1000

FRAUD_RATE = 0.10

OUTPUT_FILE = "synthetic_data/synthetic_upi_transactions.csv"


# ============================================================
# SAMPLE DATA
# ============================================================

TRANSACTION_TYPES = [
    "P2P",
    "P2M",
    "QR"
]

MERCHANT_CATEGORIES = [
    "Grocery",
    "Restaurant",
    "Shopping",
    "Fuel",
    "Travel",
    "Entertainment",
    "Utilities"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def generate_ip():

    return (
        f"{random.randint(1, 255)}."
        f"{random.randint(0, 255)}."
        f"{random.randint(0, 255)}."
        f"{random.randint(1, 255)}"
    )


def generate_transaction_id():

    return "TXN" + str(uuid.uuid4())[:8].upper()


def generate_user_id():

    return "USER" + str(random.randint(1000, 9999))


def generate_receiver_id():

    return "USER" + str(random.randint(1000, 9999))


def generate_device_id():

    return "DEV" + str(random.randint(1000, 9999))


def generate_merchant_id():

    return "MER" + str(random.randint(1000, 9999))


# ============================================================
# BASE LEGITIMATE TRANSACTION
# ============================================================

def generate_legitimate_transaction():

    transaction_amount = round(
        random.uniform(100, 10000),
        2
    )

    average_transaction_amount = round(
        random.uniform(500, 5000),
        2
    )

    sender_balance_before = round(
        random.uniform(10000, 100000),
        2
    )

    sender_balance_after = round(
        max(
            0,
            sender_balance_before - transaction_amount
        ),
        2
    )

    transaction_type = random.choice(
        TRANSACTION_TYPES
    )

    return {

        "transaction_id":
            generate_transaction_id(),

        "user_id":
            generate_user_id(),

        "transaction_amount":
            transaction_amount,

        "transaction_type":
            transaction_type,

        "timestamp":
            datetime.now() -
            timedelta(
                minutes=random.randint(
                    0,
                    100000
                )
            ),

        "sender_balance_before":
            sender_balance_before,

        "sender_balance_after":
            sender_balance_after,

        "receiver_id":
            generate_receiver_id(),

        "is_new_receiver":
            random.choice([0, 0, 0, 1]),

        "device_id":
            generate_device_id(),

        "is_new_device":
            random.choice([0, 0, 0, 1]),

        "latitude":
            round(
                random.uniform(
                    18.40,
                    18.65
                ),
                6
            ),

        "longitude":
            round(
                random.uniform(
                    73.70,
                    74.00
                ),
                6
            ),

        "distance_from_usual_location":
            round(
                random.uniform(
                    0,
                    20
                ),
                2
            ),

        "ip_address":
            generate_ip(),

        "is_new_ip":
            random.choice([0, 0, 0, 1]),

        "transactions_last_1_hour":
            random.randint(0, 4),

        "transactions_last_24_hours":
            random.randint(1, 20),

        "average_transaction_amount":
            average_transaction_amount,

        "amount_deviation_ratio":
            round(
                transaction_amount /
                max(
                    average_transaction_amount,
                    1
                ),
                2
            ),

        "failed_transaction_count":
            random.randint(0, 1),

        "qr_transaction":
            1 if transaction_type == "QR"
            else 0,

        "merchant_id":
            generate_merchant_id(),

        "merchant_category":
            random.choice(
                MERCHANT_CATEGORIES
            ),

        "merchant_risk_score":
            round(
                random.uniform(
                    0.00,
                    0.30
                ),
                2
            ),

        "account_age_days":
            random.randint(
                30,
                2000
            ),

        "fraud_scenario":
            "LEGITIMATE",

        "is_fraud":
            0
    }


# ============================================================
# FRAUD SCENARIO 1
# HIGH VALUE TRANSACTION
# ============================================================

def generate_high_value_fraud():

    transaction = generate_legitimate_transaction()

    transaction["transaction_amount"] = round(
        random.uniform(
            50000,
            500000
        ),
        2
    )

    transaction["amount_deviation_ratio"] = round(
        transaction["transaction_amount"] /
        transaction["average_transaction_amount"],
        2
    )

    transaction["fraud_scenario"] = (
        "HIGH_VALUE"
    )

    transaction["is_fraud"] = 1

    return transaction


# ============================================================
# FRAUD SCENARIO 2
# NEW DEVICE
# ============================================================

def generate_new_device_fraud():

    transaction = generate_legitimate_transaction()

    transaction["is_new_device"] = 1

    transaction["is_new_ip"] = 1

    transaction["transaction_amount"] = round(
        random.uniform(
            10000,
            100000
        ),
        2
    )

    transaction["fraud_scenario"] = (
        "NEW_DEVICE"
    )

    transaction["is_fraud"] = 1

    return transaction


# ============================================================
# FRAUD SCENARIO 3
# NEW LOCATION
# ============================================================

def generate_new_location_fraud():

    transaction = generate_legitimate_transaction()

    transaction[
        "distance_from_usual_location"
    ] = round(
        random.uniform(
            100,
            1000
        ),
        2
    )

    transaction["transaction_amount"] = round(
        random.uniform(
            10000,
            100000
        ),
        2
    )

    transaction["fraud_scenario"] = (
        "NEW_LOCATION"
    )

    transaction["is_fraud"] = 1

    return transaction


# ============================================================
# FRAUD SCENARIO 4
# NEW RECEIVER
# ============================================================

def generate_new_receiver_fraud():

    transaction = generate_legitimate_transaction()

    transaction["is_new_receiver"] = 1

    transaction["transaction_amount"] = round(
        random.uniform(
            15000,
            100000
        ),
        2
    )

    transaction["fraud_scenario"] = (
        "NEW_RECEIVER"
    )

    transaction["is_fraud"] = 1

    return transaction


# ============================================================
# FRAUD SCENARIO 5
# HIGH TRANSACTION VELOCITY
# ============================================================

def generate_velocity_fraud():

    transaction = generate_legitimate_transaction()

    transaction[
        "transactions_last_1_hour"
    ] = random.randint(
        8,
        20
    )

    transaction[
        "transactions_last_24_hours"
    ] = random.randint(
        20,
        50
    )

    transaction["transaction_amount"] = round(
        random.uniform(
            5000,
            75000
        ),
        2
    )

    transaction["fraud_scenario"] = (
        "HIGH_VELOCITY"
    )

    transaction["is_fraud"] = 1

    return transaction


# ============================================================
# FRAUD SCENARIO 6
# MULTIPLE FAILED ATTEMPTS
# ============================================================

def generate_failed_attempt_fraud():

    transaction = generate_legitimate_transaction()

    transaction[
        "failed_transaction_count"
    ] = random.randint(
        3,
        8
    )

    transaction["transaction_amount"] = round(
        random.uniform(
            10000,
            100000
        ),
        2
    )

    transaction["fraud_scenario"] = (
        "FAILED_ATTEMPTS"
    )

    transaction["is_fraud"] = 1

    return transaction


# ============================================================
# FRAUD SCENARIO 7
# QR FRAUD
# ============================================================

def generate_qr_fraud():

    transaction = generate_legitimate_transaction()

    transaction["transaction_type"] = "QR"

    transaction["qr_transaction"] = 1

    transaction["is_new_receiver"] = 1

    transaction["transaction_amount"] = round(
        random.uniform(
            10000,
            100000
        ),
        2
    )

    transaction["merchant_risk_score"] = round(
        random.uniform(
            0.60,
            1.00
        ),
        2
    )

    transaction["fraud_scenario"] = (
        "QR_FRAUD"
    )

    transaction["is_fraud"] = 1

    return transaction


# ============================================================
# FRAUD SCENARIO 8
# MERCHANT FRAUD
# ============================================================

def generate_merchant_fraud():

    transaction = generate_legitimate_transaction()

    transaction["merchant_risk_score"] = round(
        random.uniform(
            0.70,
            1.00
        ),
        2
    )

    transaction["transaction_amount"] = round(
        random.uniform(
            20000,
            150000
        ),
        2
    )

    transaction["fraud_scenario"] = (
        "MERCHANT_FRAUD"
    )

    transaction["is_fraud"] = 1

    return transaction


# ============================================================
# FRAUD SCENARIO 9
# ACCOUNT TAKEOVER
# ============================================================

def generate_account_takeover_fraud():

    transaction = generate_legitimate_transaction()

    transaction["is_new_device"] = 1

    transaction["is_new_ip"] = 1

    transaction["is_new_receiver"] = 1

    transaction[
        "distance_from_usual_location"
    ] = round(
        random.uniform(
            200,
            1000
        ),
        2
    )

    transaction[
        "transactions_last_1_hour"
    ] = random.randint(
        5,
        15
    )

    transaction[
        "failed_transaction_count"
    ] = random.randint(
        2,
        6
    )

    transaction["transaction_amount"] = round(
        random.uniform(
            25000,
            200000
        ),
        2
    )

    transaction["fraud_scenario"] = (
        "ACCOUNT_TAKEOVER"
    )

    transaction["is_fraud"] = 1

    return transaction


# ============================================================
# FRAUD SCENARIO 10
# MULTI-SIGNAL FRAUD
# ============================================================

def generate_multi_signal_fraud():

    transaction = generate_legitimate_transaction()

    transaction["is_new_device"] = 1

    transaction["is_new_ip"] = 1

    transaction["is_new_receiver"] = 1

    transaction[
        "distance_from_usual_location"
    ] = round(
        random.uniform(
            300,
            1500
        ),
        2
    )

    transaction[
        "transactions_last_1_hour"
    ] = random.randint(
        10,
        25
    )

    transaction[
        "transactions_last_24_hours"
    ] = random.randint(
        30,
        60
    )

    transaction[
        "failed_transaction_count"
    ] = random.randint(
        4,
        10
    )

    transaction["transaction_amount"] = round(
        random.uniform(
            50000,
            500000
        ),
        2
    )

    transaction["qr_transaction"] = 1

    transaction["transaction_type"] = "QR"

    transaction["merchant_risk_score"] = round(
        random.uniform(
            0.70,
            1.00
        ),
        2
    )

    transaction["fraud_scenario"] = (
        "MULTI_SIGNAL"
    )

    transaction["is_fraud"] = 1

    return transaction


# ============================================================
# GENERATE DATASET
# ============================================================

def generate_dataset():

    transactions = []

    legitimate_count = int(
        NUMBER_OF_TRANSACTIONS *
        (1 - FRAUD_RATE)
    )

    fraud_count = (
        NUMBER_OF_TRANSACTIONS -
        legitimate_count
    )

    # -----------------------------------------
    # Legitimate transactions
    # -----------------------------------------

    for _ in range(
        legitimate_count
    ):

        transactions.append(
            generate_legitimate_transaction()
        )

    # -----------------------------------------
    # Fraud transactions
    # -----------------------------------------

    fraud_generators = [

        generate_high_value_fraud,

        generate_new_device_fraud,

        generate_new_location_fraud,

        generate_new_receiver_fraud,

        generate_velocity_fraud,

        generate_failed_attempt_fraud,

        generate_qr_fraud,

        generate_merchant_fraud,

        generate_account_takeover_fraud,

        generate_multi_signal_fraud

    ]

    for _ in range(fraud_count):

        generator = random.choice(
            fraud_generators
        )

        transactions.append(
            generator()
        )

    # Shuffle dataset

    random.shuffle(
        transactions
    )

    return pd.DataFrame(
        transactions
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n======================================"
    )

    print(
        "FINSECURE AI"
    )

    print(
        "Synthetic UPI Dataset Generator"
    )

    print(
        "======================================"
    )

    print(
        "\nGenerating dataset..."
    )

    df = generate_dataset()

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\nDataset generated successfully!"
    )

    print(
        "Total transactions:",
        len(df)
    )

    print(
        "\nFraud distribution:"
    )

    print(
        df["is_fraud"].value_counts()
    )

    print(
        "\nFraud percentage:"
    )

    print(
        df["is_fraud"]
        .value_counts(
            normalize=True
        ) * 100
    )

    print(
        "\nFraud scenario distribution:"
    )

    print(
        df[
            df["is_fraud"] == 1
        ]["fraud_scenario"]
        .value_counts()
    )

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\n======================================"
    )