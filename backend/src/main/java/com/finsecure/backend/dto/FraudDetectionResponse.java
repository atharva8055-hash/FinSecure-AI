package com.finsecure.backend.dto;

public class FraudDetectionResponse {

    private String status;
    private Prediction prediction;

    public FraudDetectionResponse() {
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public Prediction getPrediction() {
        return prediction;
    }

    public void setPrediction(Prediction prediction) {
        this.prediction = prediction;
    }

    public static class Prediction {

        private double fraud_probability;
        private double risk_score;
        private String risk_level;
        private boolean is_fraud;
        private String decision;

        public Prediction() {
        }

        public double getFraud_probability() {
            return fraud_probability;
        }

        public void setFraud_probability(double fraud_probability) {
            this.fraud_probability = fraud_probability;
        }

        public double getRisk_score() {
            return risk_score;
        }

        public void setRisk_score(double risk_score) {
            this.risk_score = risk_score;
        }

        public String getRisk_level() {
            return risk_level;
        }

        public void setRisk_level(String risk_level) {
            this.risk_level = risk_level;
        }

        public boolean isIs_fraud() {
            return is_fraud;
        }

        public void setIs_fraud(boolean is_fraud) {
            this.is_fraud = is_fraud;
        }

        public String getDecision() {
            return decision;
        }

        public void setDecision(String decision) {
            this.decision = decision;
        }
    }
}