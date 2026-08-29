import pandas as pd
import os

print("\n========== MODEL COMPARISON ==========\n")


# ============================================================
# HELPER FUNCTION
# ============================================================

def load_result_file(path):
    """
    Loads a model result CSV and prints its structure.
    """

    if not os.path.exists(path):
        print(f"WARNING: File not found -> {path}")
        return None

    df = pd.read_csv(path)

    print(f"Loaded: {path}")
    print("Columns:", list(df.columns))

    return df


def get_metric(df, metric_name):
    """
    Extract a metric from either:
    1. metric/value format
    2. single-row model result format
    """

    if df is None:
        return None

    # --------------------------------------------------------
    # Format 1:
    # metric | value
    # --------------------------------------------------------

    if "metric" in df.columns and "value" in df.columns:

        row = df[df["metric"] == metric_name]

        if len(row) > 0:
            return float(row.iloc[0]["value"])

    # --------------------------------------------------------
    # Format 2:
    # Metric directly represented as a column
    # --------------------------------------------------------

    if metric_name in df.columns:

        return float(df.iloc[0][metric_name])

    return None


# ============================================================
# 1. LOAD LOGISTIC REGRESSION RESULTS
# ============================================================

print("========== LOADING LOGISTIC REGRESSION RESULTS ==========")

baseline_path = "dataset/baseline_model_results.csv"

baseline = load_result_file(baseline_path)


# ============================================================
# 2. LOAD RANDOM FOREST V2 RESULTS
# ============================================================

print("\n========== LOADING RANDOM FOREST V2 RESULTS ==========")

rf_v2_path = "dataset/random_forest_v2_results.csv"

rf_v2 = load_result_file(rf_v2_path)


# ============================================================
# 3. LOAD LEAKAGE-SAFE RESULTS
# ============================================================

print("\n========== LOADING LEAKAGE-SAFE RESULTS ==========")

leakage_safe_path = (
    "dataset/leakage_safe_validation_results.csv"
)

leakage_safe = load_result_file(
    leakage_safe_path
)


# ============================================================
# 4. LEAKAGE-SAFE METRICS
# ============================================================

print("\n========== LEAKAGE-SAFE RESULTS ==========")

ls_roc = get_metric(
    leakage_safe,
    "ROC-AUC"
)

ls_pr = get_metric(
    leakage_safe,
    "PR-AUC"
)

ls_fp = get_metric(
    leakage_safe,
    "False Positives"
)

ls_fn = get_metric(
    leakage_safe,
    "False Negatives"
)

ls_tp = get_metric(
    leakage_safe,
    "True Positives"
)

ls_tn = get_metric(
    leakage_safe,
    "True Negatives"
)


ls_precision = (
    ls_tp / (ls_tp + ls_fp)
    if ls_tp is not None
    and ls_fp is not None
    and (ls_tp + ls_fp) > 0
    else None
)

ls_recall = (
    ls_tp / (ls_tp + ls_fn)
    if ls_tp is not None
    and ls_fn is not None
    and (ls_tp + ls_fn) > 0
    else None
)

ls_f1 = (
    2 * ls_precision * ls_recall /
    (ls_precision + ls_recall)
    if ls_precision is not None
    and ls_recall is not None
    and (ls_precision + ls_recall) > 0
    else None
)


print("ROC-AUC :", ls_roc)
print("PR-AUC  :", ls_pr)
print("Precision:", ls_precision)
print("Recall   :", ls_recall)
print("F1-score :", ls_f1)
print("False Positives:", ls_fp)
print("False Negatives:", ls_fn)
print("True Positives :", ls_tp)


# ============================================================
# 5. CREATE COMPARISON
# ============================================================

comparison = []


# ============================================================
# 6. RANDOM FOREST V2
# ============================================================

if rf_v2 is not None:

    rf_roc = get_metric(
        rf_v2,
        "ROC-AUC"
    )

    rf_pr = get_metric(
        rf_v2,
        "PR-AUC"
    )

    rf_fp = get_metric(
        rf_v2,
        "False Positives"
    )

    rf_fn = get_metric(
        rf_v2,
        "False Negatives"
    )

    rf_tp = get_metric(
        rf_v2,
        "True Positives"
    )

    rf_tn = get_metric(
        rf_v2,
        "True Negatives"
    )

    if (
        rf_tp is not None
        and rf_fp is not None
    ):
        rf_precision = (
            rf_tp / (rf_tp + rf_fp)
        )
    else:
        rf_precision = None

    if (
        rf_tp is not None
        and rf_fn is not None
    ):
        rf_recall = (
            rf_tp / (rf_tp + rf_fn)
        )
    else:
        rf_recall = None

    if (
        rf_precision is not None
        and rf_recall is not None
        and (rf_precision + rf_recall) > 0
    ):
        rf_f1 = (
            2 * rf_precision * rf_recall /
            (rf_precision + rf_recall)
        )
    else:
        rf_f1 = None

    comparison.append({
        "Model": "Random Forest V2",
        "Validation": "Random Split",
        "Features": 17,
        "ROC-AUC": rf_roc,
        "PR-AUC": rf_pr,
        "Precision": rf_precision,
        "Recall": rf_recall,
        "F1-score": rf_f1,
        "False Positives": rf_fp,
        "False Negatives": rf_fn,
        "True Positives": rf_tp
    })


# ============================================================
# 7. LOGISTIC REGRESSION
# ============================================================

if baseline is not None:

    baseline_roc = get_metric(
        baseline,
        "ROC-AUC"
    )

    baseline_pr = get_metric(
        baseline,
        "PR-AUC"
    )

    baseline_fp = get_metric(
        baseline,
        "False Positives"
    )

    baseline_fn = get_metric(
        baseline,
        "False Negatives"
    )

    baseline_tp = get_metric(
        baseline,
        "True Positives"
    )

    if (
        baseline_tp is not None
        and baseline_fp is not None
    ):
        baseline_precision = (
            baseline_tp /
            (baseline_tp + baseline_fp)
        )
    else:
        baseline_precision = None

    if (
        baseline_tp is not None
        and baseline_fn is not None
    ):
        baseline_recall = (
            baseline_tp /
            (baseline_tp + baseline_fn)
        )
    else:
        baseline_recall = None

    if (
        baseline_precision is not None
        and baseline_recall is not None
        and (baseline_precision + baseline_recall) > 0
    ):
        baseline_f1 = (
            2 * baseline_precision * baseline_recall /
            (baseline_precision + baseline_recall)
        )
    else:
        baseline_f1 = None

    comparison.append({
        "Model": "Logistic Regression",
        "Validation": "Random Split",
        "Features": 19,
        "ROC-AUC": baseline_roc,
        "PR-AUC": baseline_pr,
        "Precision": baseline_precision,
        "Recall": baseline_recall,
        "F1-score": baseline_f1,
        "False Positives": baseline_fp,
        "False Negatives": baseline_fn,
        "True Positives": baseline_tp
    })


# ============================================================
# 8. LEAKAGE-SAFE RANDOM FOREST
# ============================================================

comparison.append({
    "Model": "Leakage-Safe Random Forest",
    "Validation": "Time-Aware",
    "Features": 9,
    "ROC-AUC": ls_roc,
    "PR-AUC": ls_pr,
    "Precision": ls_precision,
    "Recall": ls_recall,
    "F1-score": ls_f1,
    "False Positives": ls_fp,
    "False Negatives": ls_fn,
    "True Positives": ls_tp
})


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

comparison_df = pd.DataFrame(comparison)

print("\n========== MODEL COMPARISON TABLE ==========\n")

print(
    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

output_file = (
    "dataset/model_comparison_leakage_safe.csv"
)

comparison_df.to_csv(
    output_file,
    index=False
)

print("\nComparison saved to:")
print(output_file)


# ============================================================
# 11. INTERPRETATION
# ============================================================

print("\n========== INTERPRETATION ==========")

print("""
The Leakage-Safe Random Forest uses only features that
are available at transaction decision time.

The model also uses a time-aware validation strategy,
where earlier transactions are used for training and
later transactions are used for testing.

The previous Random Forest V2 used additional balance
features and random-split validation.

Therefore, the leakage-safe model provides the more
conservative estimate of real-world performance.
""")


# ============================================================
# 12. RECOMMENDATION
# ============================================================

print("\n========== MODEL RECOMMENDATION ==========")

print("""
The Leakage-Safe Random Forest should be treated as the
primary candidate for FinSecure AI.

Before integrating it into the final fraud detection
pipeline, threshold optimization should be performed.

The objective will be to reduce false positives while
maintaining strong fraud recall.
""")


print("\n========================================")
print("MODEL COMPARISON COMPLETE")
print("========================================")

print("\nNext step:")
print("Leakage-safe threshold analysis.")