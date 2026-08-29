package com.finsecure.backend.entity;

import jakarta.persistence.*;

import java.time.LocalDateTime;

@Entity
@Table(name = "fraud_transactions")
public class FraudTransaction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    private int step;

    private double amount;

    private double oldbalanceOrg;

    private double oldbalanceDest;

    private String type;

    private double fraudProbability;

    private double riskScore;

    private String riskLevel;

    private boolean fraud;

    private String decision;

    private LocalDateTime createdAt;

    public FraudTransaction() {
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public User getUser() {
        return user;
    }

    public void setUser(User user) {
        this.user = user;
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

    public double getFraudProbability() {
        return fraudProbability;
    }

    public void setFraudProbability(double fraudProbability) {
        this.fraudProbability = fraudProbability;
    }

    public double getRiskScore() {
        return riskScore;
    }

    public void setRiskScore(double riskScore) {
        this.riskScore = riskScore;
    }

    public String getRiskLevel() {
        return riskLevel;
    }

    public void setRiskLevel(String riskLevel) {
        this.riskLevel = riskLevel;
    }

    public boolean isFraud() {
        return fraud;
    }

    public void setFraud(boolean fraud) {
        this.fraud = fraud;
    }

    public String getDecision() {
        return decision;
    }

    public void setDecision(String decision) {
        this.decision = decision;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }
}
