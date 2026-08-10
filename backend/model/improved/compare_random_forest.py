import pandas as pd


print("\n========== RANDOM FOREST MODEL COMPARISON ==========")


# ============================================================
# LOAD MODEL RESULTS
# ============================================================

v1_path = "dataset/random_forest_results.csv"
v2_path = "dataset/random_forest_v2_results.csv"

print("\nLoading Random Forest V1 results...")
v1 = pd.read_csv(v1_path)

print("Loading Random Forest V2 results...")
v2 = pd.read_csv(v2_path)


# ============================================================
# EXTRACT RESULTS
# ============================================================

v1_row = v1.iloc[0]
v2_row = v2.iloc[0]


# ============================================================
# CREATE COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({
    "Metric": [
        "Number of Features",
        "ROC-AUC",
        "PR-AUC",
        "True Negatives",
        "False Positives",
        "False Negatives",
        "True Positives"
    ],

    "Random Forest V1": [
        19,
        v1_row["roc_auc"],
        v1_row["pr_auc"],
        v1_row["true_negatives"],
        v1_row["false_positives"],
        v1_row["false_negatives"],
        v1_row["true_positives"]
    ],

    "Random Forest V2": [
        17,
        v2_row["roc_auc"],
        v2_row["pr_auc"],
        v2_row["true_negatives"],
        v2_row["false_positives"],
        v2_row["false_negatives"],
        v2_row["true_positives"]
    ]
})


# ============================================================
# DISPLAY COMPARISON
# ============================================================

print("\n========== V1 VS V2 ==========")

print(
    comparison.to_string(index=False)
)


# ============================================================
# DIFFERENCE ANALYSIS
# ============================================================

roc_difference = (
    v2_row["roc_auc"] -
    v1_row["roc_auc"]
)

pr_difference = (
    v2_row["pr_auc"] -
    v1_row["pr_auc"]
)

fp_difference = (
    v2_row["false_positives"] -
    v1_row["false_positives"]
)

fn_difference = (
    v2_row["false_negatives"] -
    v1_row["false_negatives"]
)


print("\n========== DIFFERENCE ANALYSIS ==========")

print(
    "ROC-AUC difference:",
    roc_difference
)

print(
    "PR-AUC difference:",
    pr_difference
)

print(
    "False Positive difference:",
    fp_difference
)

print(
    "False Negative difference:",
    fn_difference
)


# ============================================================
# MODEL INTERPRETATION
# ============================================================

print("\n========== MODEL INTERPRETATION ==========")

if v2_row["pr_auc"] >= v1_row["pr_auc"]:
    print("V2 maintains or improves PR-AUC.")
else:
    print("V2 has slightly lower PR-AUC.")

if v2_row["false_negatives"] <= v1_row["false_negatives"]:
    print("V2 maintains or improves fraud detection recall.")
else:
    print("V2 has more missed fraud cases.")

if v2_row["false_positives"] < v1_row["false_positives"]:
    print("V2 reduces false positives.")
elif v2_row["false_positives"] > v1_row["false_positives"]:
    print("V2 increases false positives.")
else:
    print("Both models have the same false positives.")


# ============================================================
# RECOMMENDATION
# ============================================================

print("\n========== MODEL RECOMMENDATION ==========")

if (
    v2_row["false_negatives"] <= v1_row["false_negatives"]
    and
    v2_row["pr_auc"] >= v1_row["pr_auc"] * 0.999
):

    print("Recommended model: Random Forest V2")

    print(
        "Reason: V2 achieves comparable predictive performance "
        "with fewer features."
    )

else:

    print("Recommended model: Random Forest V1")

    print(
        "Reason: V1 provides stronger overall predictive performance."
    )


# ============================================================
# SAVE COMPARISON
# ============================================================

output_path = "dataset/random_forest_comparison.csv"

comparison.to_csv(
    output_path,
    index=False
)

print("\nComparison saved to:")
print(output_path)


print("\n========================================")
print("RANDOM FOREST COMPARISON COMPLETE")
print("========================================")

print("\nNext step:")
print("Time-aware validation of the selected model.")