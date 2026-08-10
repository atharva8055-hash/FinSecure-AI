import pandas as pd


print("\n========== VALIDATION COMPARISON ==========")


# ============================================================
# LOAD RESULTS
# ============================================================

random_results = pd.read_csv(
    "dataset/random_forest_v2_results.csv"
)

time_results = pd.read_csv(
    "dataset/time_aware_validation_results.csv"
)


# ============================================================
# EXTRACT RESULTS
# ============================================================

random_row = random_results.iloc[0]
time_row = time_results.iloc[0]


# ============================================================
# COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({
    "Metric": [
        "Validation Type",
        "ROC-AUC",
        "PR-AUC",
        "True Negatives",
        "False Positives",
        "False Negatives",
        "True Positives"
    ],

    "Random Split": [
        "Random 80/20",
        random_row["roc_auc"],
        random_row["pr_auc"],
        random_row["true_negatives"],
        random_row["false_positives"],
        random_row["false_negatives"],
        random_row["true_positives"]
    ],

    "Time-Aware": [
        "Past → Future",
        time_row["roc_auc"],
        time_row["pr_auc"],
        time_row["true_negatives"],
        time_row["false_positives"],
        time_row["false_negatives"],
        time_row["true_positives"]
    ]
})


# ============================================================
# DISPLAY
# ============================================================

print("\n========== RANDOM SPLIT VS TIME-AWARE ==========")

print(
    comparison.to_string(index=False)
)


# ============================================================
# DIFFERENCES
# ============================================================

print("\n========== PERFORMANCE DIFFERENCES ==========")

print(
    "ROC-AUC difference:",
    time_row["roc_auc"] - random_row["roc_auc"]
)

print(
    "PR-AUC difference:",
    time_row["pr_auc"] - random_row["pr_auc"]
)

print(
    "False Positive difference:",
    time_row["false_positives"] -
    random_row["false_positives"]
)

print(
    "False Negative difference:",
    time_row["false_negatives"] -
    random_row["false_negatives"]
)


# ============================================================
# INTERPRETATION
# ============================================================

print("\n========== INTERPRETATION ==========")

print(
    "Random-split validation evaluates general predictive "
    "performance using randomly selected transactions."
)

print(
    "Time-aware validation evaluates whether a model trained "
    "on earlier transaction periods can detect fraud in later periods."
)

if time_row["pr_auc"] >= random_row["pr_auc"]:
    print(
        "Time-aware PR-AUC is equal to or higher than "
        "random-split PR-AUC."
    )
else:
    print(
        "Time-aware PR-AUC is lower than random-split PR-AUC."
    )


# ============================================================
# PERFECT PERFORMANCE WARNING
# ============================================================

if (
    time_row["roc_auc"] == 1.0
    and
    time_row["pr_auc"] == 1.0
    and
    time_row["false_positives"] == 0
    and
    time_row["false_negatives"] == 0
):

    print("\nWARNING:")
    print(
        "The time-aware model achieved perfect performance."
    )

    print(
        "This should be investigated for potential "
        "dataset-specific separability or feature leakage "
        "before claiming production-level performance."
    )


# ============================================================
# SAVE
# ============================================================

output_path = "dataset/validation_comparison.csv"

comparison.to_csv(
    output_path,
    index=False
)

print("\nComparison saved to:")
print(output_path)


print("\n========================================")
print("VALIDATION COMPARISON COMPLETE")
print("========================================")

print("\nNext step:")
print(
    "Stress-test Random Forest V2 by removing "
    "post-transaction balance features."
)