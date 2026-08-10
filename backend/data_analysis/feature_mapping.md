# FinSecure AI — UPI Fraud Detection
# Feature Mapping: PaySim → UPI

## 1. Transaction Features

| PaySim Feature | FinSecure Feature | Type | Purpose |
|---|---|---|---|
| amount | transaction_amount | float | Amount of UPI transaction |
| type | transaction_type | categorical | Type of UPI transaction |
| step | transaction_time | integer/time | Time of transaction |

---

## 2. Sender Features

| PaySim Feature | FinSecure Feature | Type | Purpose |
|---|---|---|---|
| nameOrig | sender_id | categorical | Unique sender identifier |
| oldbalanceOrg | sender_balance_before | float | Sender balance before transaction |
| newbalanceOrig | sender_balance_after | float | Sender balance after transaction |

### Derived Sender Features

| Feature | Purpose |
|---|---|
| sender_balance_change | Detect significant balance reduction |
| sender_balance_change_ratio | Measure percentage of balance reduction |
| sender_average_transaction | Establish normal spending behavior |
| sender_transaction_count | Measure transaction frequency |
| sender_recent_transaction_count | Detect transaction bursts |

---

## 3. Receiver Features

| PaySim Feature | FinSecure Feature | Type | Purpose |
|---|---|---|---|
| nameDest | receiver_id | categorical | Receiver identifier |
| oldbalanceDest | receiver_balance_before | float | Receiver balance before transaction |
| newbalanceDest | receiver_balance_after | float | Receiver balance after transaction |

### Derived Receiver Features

| Feature | Purpose |
|---|---|
| receiver_balance_change | Detect unusual incoming funds |
| receiver_transaction_count | Identify frequently used receivers |
| is_new_receiver | Detect first-time receiver |
| receiver_frequency | Measure receiver interaction frequency |

---

## 4. Temporal Features

PaySim provides `step`.

For FinSecure AI we will eventually generate:

| Feature | Purpose |
|---|---|
| transaction_hour | Identify unusual transaction times |
| transaction_day | Identify unusual transaction days |
| transactions_last_1_hour | Detect transaction bursts |
| transactions_last_24_hours | Detect unusual daily activity |
| time_since_last_transaction | Detect rapid transactions |

---

## 5. Behavioral Features

These features are extremely important for UPI fraud detection.

| Feature | Purpose |
|---|---|
| average_transaction_amount | User's normal transaction value |
| amount_deviation | Detect unusually large transactions |
| previous_transaction_amount | Compare with previous transaction |
| transaction_velocity | Detect rapid transactions |
| unique_receivers_count | Detect unusual receiver activity |
| new_receiver | Detect first-time payment recipient |
| failed_transaction_count | Identify suspicious repeated attempts |

---

## 6. Device Features

PaySim does not provide device information.

These will therefore be introduced in Stage 2.

| Feature | Purpose |
|---|---|
| device_id | Identify user's device |
| is_new_device | Detect transactions from unfamiliar device |
| device_transaction_count | Measure device activity |
| device_change_frequency | Detect frequent device changes |

---

## 7. Location Features

PaySim does not provide UPI location information.

Stage 2 will introduce:

| Feature | Purpose |
|---|---|
| latitude | Transaction location |
| longitude | Transaction location |
| location_change | Detect unusual location |
| distance_from_usual_location | Detect geographically unusual transaction |

---

## 8. Network Features

Stage 2 will introduce:

| Feature | Purpose |
|---|---|
| ip_address | Source network identification |
| is_new_ip | Detect unfamiliar network |
| ip_transaction_count | Measure network activity |

---

## 9. UPI / Merchant Features

Stage 2 will introduce:

| Feature | Purpose |
|---|---|
| upi_id | UPI identifier |
| merchant_id | Merchant identifier |
| merchant_category | Merchant classification |
| qr_transaction | Identify QR-based transaction |
| merchant_risk_score | Merchant-level risk |

---

## 10. Fraud Target

| PaySim Feature | FinSecure Feature | Purpose |
|---|---|---|
| isFraud | is_fraud | Model target |

`is_fraud` will be:

0 = Legitimate

1 = Fraudulent

---

## 11. PaySim Feature Not Used Directly

### isFlaggedFraud

This feature will NOT be used as a primary model feature.

Reason:

Only 16 transactions were marked with:

isFlaggedFraud = 1

while there were 8,213 actual fraudulent transactions.

Therefore, FinSecure AI should learn fraud patterns independently rather than depending on an existing fraud flag.