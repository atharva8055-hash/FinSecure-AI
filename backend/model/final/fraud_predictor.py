import os
import joblib
import pandas as pd


# ============================================================
# FINSECURE AI - FRAUD PREDICTION INFERENCE MODULE
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "finsecure_leakage_safe_random_forest.joblib"
)

FINAL_THRESHOLD = 0.5

FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER"
]

TRANSACTION_TYPES = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER"
]


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    print("Final FinSecure Random Forest loaded successfully.")

    return model


# ============================================================
# PREPARE TRANSACTION
# ============================================================

def prepare_transaction(transaction):
    """
    Convert a single transaction into the exact
    9-feature format used during final model training.
    """

    required_fields = [
        "step",
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
        "type"
    ]

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    for field in required_fields:
        if field not in transaction:
            raise ValueError(
                f"Missing required field: {field}"
            )

    transaction_type = str(
        transaction["type"]
    ).upper()

    if transaction_type not in TRANSACTION_TYPES:
        raise ValueError(
            f"Invalid transaction type: {transaction_type}"
        )

    # --------------------------------------------------------
    # Create base feature row
    # --------------------------------------------------------

    row = {
        "step": float(transaction["step"]),
        "amount": float(transaction["amount"]),
        "oldbalanceOrg": float(transaction["oldbalanceOrg"]),
        "oldbalanceDest": float(transaction["oldbalanceDest"])
    }

    # --------------------------------------------------------
    # One-hot encode transaction type
    # --------------------------------------------------------

    for tx_type in TRANSACTION_TYPES:
        feature_name = f"type_{tx_type}"

        row[feature_name] = (
            1 if transaction_type == tx_type else 0
        )

    # --------------------------------------------------------
    # Create DataFrame in EXACT feature order
    # --------------------------------------------------------

    X = pd.DataFrame(
        [row],
        columns=FEATURES
    )

    return X


# ============================================================
# PREDICT FRAUD
# ============================================================

def predict_fraud(model, transaction):
    """
    Predict whether a transaction is fraudulent.
    """

    X = prepare_transaction(transaction)

    # --------------------------------------------------------
    # Fraud probability
    # --------------------------------------------------------

    fraud_probability = model.predict_proba(X)[0][1]

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    is_fraud = (
        fraud_probability >= FINAL_THRESHOLD
    )

    # --------------------------------------------------------
    # Risk score
    # --------------------------------------------------------
    # Convert probability into a 0-100 score.

    risk_score = round(
        fraud_probability * 100,
        2
    )

    if risk_score >= 80:
        risk_level = "HIGH"

    elif risk_score >= 50:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {
        "fraud_probability": round(
            float(fraud_probability),
            6
        ),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "is_fraud": bool(is_fraud),
        "decision": (
            "FRAUD"
            if is_fraud
            else "LEGITIMATE"
        )
    }

    return result


# ============================================================
# TEST FUNCTION
# ============================================================

def main():

    print()
    print("=" * 60)
    print("FINSECURE AI - FRAUD PREDICTION INFERENCE")
    print("=" * 60)

    print()
    print("Loading final model...")

    model = load_model()

    # --------------------------------------------------------
    # Sample legitimate transaction
    # --------------------------------------------------------

    legitimate_transaction = {
        "step": 100,
        "amount": 5000,
        "oldbalanceOrg": 25000,
        "oldbalanceDest": 10000,
        "type": "PAYMENT"
    }

    print()
    print("========== TEST TRANSACTION ==========")

    print(
        "Transaction type:",
        legitimate_transaction["type"]
    )

    print(
        "Amount:",
        legitimate_transaction["amount"]
    )

    print(
        "Sender balance:",
        legitimate_transaction["oldbalanceOrg"]
    )

    print(
        "Receiver balance:",
        legitimate_transaction["oldbalanceDest"]
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    result = predict_fraud(
        model,
        legitimate_transaction
    )

    print()
    print("========== PREDICTION RESULT ==========")

    print(
        "Fraud probability:",
        result["fraud_probability"]
    )

    print(
        "Risk score:",
        result["risk_score"]
    )

    print(
        "Risk level:",
        result["risk_level"]
    )

    print(
        "Decision:",
        result["decision"]
    )

    print(
        "Is fraud:",
        result["is_fraud"]
    )

    print()
    print("=" * 60)
    print("INFERENCE MODULE TEST COMPLETE")
    print("=" * 60)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()