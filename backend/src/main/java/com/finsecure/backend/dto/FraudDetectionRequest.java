package com.finsecure.backend.dto;

public class FraudDetectionRequest {

    private int step;
    private double amount;
    private double oldbalanceOrg;
    private double oldbalanceDest;
    private String type;

    public FraudDetectionRequest() {
    }

    public int getStep() {
        return step;
    }

    public void setStep(int step) {
        this.step = step;
    }

    public double getAmount() {
        return amount;
    }

    public void setAmount(double amount) {
        this.amount = amount;
    }

    public double getOldbalanceOrg() {
        return oldbalanceOrg;
    }

    public void setOldbalanceOrg(double oldbalanceOrg) {
        this.oldbalanceOrg = oldbalanceOrg;
    }

    public double getOldbalanceDest() {
        return oldbalanceDest;
    }

    public void setOldbalanceDest(double oldbalanceDest) {
        this.oldbalanceDest = oldbalanceDest;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }
}