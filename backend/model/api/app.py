from flask import Flask, request, jsonify
import sys
import os

# ============================================================
# FINSECURE AI - ML PREDICTION API
# ============================================================

app = Flask(__name__)


# ============================================================
# IMPORT FRAUD PREDICTOR
# ============================================================

FINAL_MODEL_DIRECTORY = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "final"
    )
)

sys.path.insert(0, FINAL_MODEL_DIRECTORY)

from fraud_predictor import load_model, predict_fraud


# ============================================================
# LOAD MODEL ONCE WHEN API STARTS
# ============================================================

print()
print("=" * 60)
print("FINSECURE AI - ML FRAUD DETECTION API")
print("=" * 60)

print()
print("Loading final fraud detection model...")

try:
    model = load_model()

    print("Model loaded successfully.")
    print("ML API is ready.")

except Exception as e:

    print("ERROR: Could not load final model.")
    print("Reason:", str(e))

    model = None


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    if model is None:

        return jsonify({
            "status": "ERROR",
            "message": "Fraud detection model is not loaded."
        }), 500

    return jsonify({
        "status": "OK",
        "service": "FinSecure AI Fraud Detection API",
        "model": "Leakage-Safe Random Forest",
        "model_loaded": True
    })


# ============================================================
# FRAUD PREDICTION ENDPOINT
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if model is None:

        return jsonify({
            "status": "ERROR",
            "message": "Fraud detection model is unavailable."
        }), 500

    # --------------------------------------------------------
    # Read JSON request
    # --------------------------------------------------------

    try:

        transaction = request.get_json()

    except Exception:

        return jsonify({
            "status": "ERROR",
            "message": "Invalid JSON request."
        }), 400

    # --------------------------------------------------------
    # Check request body
    # --------------------------------------------------------

    if transaction is None:

        return jsonify({
            "status": "ERROR",
            "message": "Request body is required."
        }), 400

    # --------------------------------------------------------
    # Required fields
    # --------------------------------------------------------

    required_fields = [
        "step",
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
        "type"
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in transaction
    ]

    if missing_fields:

        return jsonify({
            "status": "ERROR",
            "message": "Missing required fields.",
            "missing_fields": missing_fields
        }), 400

    # --------------------------------------------------------
    # Run prediction
    # --------------------------------------------------------

    try:

        result = predict_fraud(
            model,
            transaction
        )

    except ValueError as e:

        return jsonify({
            "status": "ERROR",
            "message": str(e)
        }), 400

    except Exception as e:

        print("Prediction error:", str(e))

        return jsonify({
            "status": "ERROR",
            "message": "Prediction failed."
        }), 500

    # --------------------------------------------------------
    # Return prediction
    # --------------------------------------------------------

    return jsonify({

        "status": "SUCCESS",

        "prediction": {

            "fraud_probability":
                result["fraud_probability"],

            "risk_score":
                result["risk_score"],

            "risk_level":
                result["risk_level"],

            "is_fraud":
                result["is_fraud"],

            "decision":
                result["decision"]
        }
    })


# ============================================================
# API INFORMATION
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({

        "application":
            "FinSecure AI",

        "service":
            "AI Fraud Detection API",

        "model":
            "Leakage-Safe Random Forest",

        "validation":
            "Time-Aware Validation",

        "features":
            9,

        "threshold":
            0.5,

        "endpoints": {

            "health":
                "GET /health",

            "prediction":
                "POST /predict"
        }
    })


# ============================================================
# START FLASK SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("STARTING FINSECURE AI ML API")
    print("=" * 60)

    print()
    print("Server:")
    print("http://127.0.0.1:5000")

    print()
    print("Health endpoint:")
    print("http://127.0.0.1:5000/health")

    print()
    print("Prediction endpoint:")
    print("http://127.0.0.1:5000/predict")

    print()
    print("Press CTRL+C to stop the server.")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )