import pandas as pd
import os


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = "dataset/leakage_safe_threshold_analysis.csv"
OUTPUT_PATH = "dataset/leakage_safe_threshold_selection.csv"

# Minimum acceptable fraud recall.
# We want to avoid missing too many fraudulent transactions.
MIN_RECALL = 0.80

# Maximum acceptable false-positive count.
# This is used as an additional practical constraint.
MAX_FALSE_POSITIVES = 25000


# ============================================================
# HELPER FUNCTION
# ============================================================

def print_threshold(row, title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print(f"Threshold        : {row['threshold']:.1f}")
    print(f"Precision        : {row['precision']:.6f}")
    print(f"Recall           : {row['recall']:.6f}")
    print(f"F1-score         : {row['f1_score']:.6f}")
    print(f"False Positives  : {int(row['false_positives'])}")
    print(f"False Negatives  : {int(row['false_negatives'])}")
    print(f"True Positives   : {int(row['true_positives'])}")
    print(f"True Negatives   : {int(row['true_negatives'])}")


# ============================================================
# LOAD RESULTS
# ============================================================

print("\n========== LEAKAGE-SAFE THRESHOLD SELECTION ==========")

print("\nLoading threshold analysis results...")

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(
        f"Threshold analysis file not found: {INPUT_PATH}"
    )

df = pd.read_csv(INPUT_PATH)

print("Threshold results loaded successfully.")

print("\nAvailable columns:")
print(list(df.columns))


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "threshold",
    "precision",
    "recall",
    "f1_score",
    "true_negatives",
    "false_positives",
    "false_negatives",
    "true_positives"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

print("\n========== THRESHOLD RESULTS ==========")

print(
    df[
        [
            "threshold",
            "precision",
            "recall",
            "f1_score",
            "false_positives",
            "false_negatives",
            "true_positives"
        ]
    ].to_string(index=False)
)


# ============================================================
# BEST F1 THRESHOLD
# ============================================================

best_f1_row = df.loc[df["f1_score"].idxmax()]

print_threshold(
    best_f1_row,
    "BEST F1 THRESHOLD"
)


# ============================================================
# HIGH-RECALL CANDIDATES
# ============================================================

print("\n========== HIGH-RECALL CANDIDATES ==========")

high_recall_df = df[
    df["recall"] >= MIN_RECALL
].copy()

if high_recall_df.empty:

    print(
        f"No threshold satisfies minimum recall of "
        f"{MIN_RECALL:.2f}"
    )

else:

    high_recall_df = high_recall_df.sort_values(
        by=["f1_score", "precision"],
        ascending=False
    )

    print(
        high_recall_df[
            [
                "threshold",
                "precision",
                "recall",
                "f1_score",
                "false_positives",
                "false_negatives",
                "true_positives"
            ]
        ].to_string(index=False)
    )


# ============================================================
# PRACTICAL CANDIDATES
# ============================================================

print("\n========== PRACTICAL THRESHOLD CANDIDATES ==========")

practical_df = df[
    (df["recall"] >= MIN_RECALL)
    &
    (df["false_positives"] <= MAX_FALSE_POSITIVES)
].copy()

if practical_df.empty:

    print(
        "No threshold satisfies both practical constraints:"
    )

    print(
        f"- Recall >= {MIN_RECALL:.2f}"
    )

    print(
        f"- False Positives <= {MAX_FALSE_POSITIVES}"
    )

else:

    practical_df = practical_df.sort_values(
        by=["f1_score", "precision"],
        ascending=False
    )

    print(
        practical_df[
            [
                "threshold",
                "precision",
                "recall",
                "f1_score",
                "false_positives",
                "false_negatives",
                "true_positives"
            ]
        ].to_string(index=False)
    )


# ============================================================
# FINAL OPERATING THRESHOLD
# ============================================================

print("\n========== FINAL OPERATING THRESHOLD SELECTION ==========")

if not practical_df.empty:

    # Select the threshold with the highest F1
    # while satisfying the recall and false-positive constraints.
    final_row = practical_df.iloc[0]

    selection_reason = (
        "Selected using practical constraints: "
        f"recall >= {MIN_RECALL:.2f} and "
        f"false positives <= {MAX_FALSE_POSITIVES}, "
        "then maximizing F1-score."
    )

else:

    # If no threshold satisfies both constraints,
    # prioritize recall and then F1.
    fallback_df = df[
        df["recall"] >= MIN_RECALL
    ].copy()

    if not fallback_df.empty:

        fallback_df = fallback_df.sort_values(
            by=["recall", "f1_score"],
            ascending=False
        )

        final_row = fallback_df.iloc[0]

        selection_reason = (
            "No threshold satisfied both practical constraints. "
            "Selected the highest-recall threshold satisfying "
            f"recall >= {MIN_RECALL:.2f}."
        )

    else:

        # Last-resort fallback: best F1.
        final_row = best_f1_row

        selection_reason = (
            "No threshold satisfied the minimum recall requirement. "
            "Fallback to the threshold with the highest F1-score."
        )


# ============================================================
# DISPLAY FINAL DECISION
# ============================================================

print_threshold(
    final_row,
    "FINAL OPERATING THRESHOLD"
)

print("\nSelection reason:")
print(selection_reason)


# ============================================================
# DEFAULT VS FINAL
# ============================================================

default_threshold = 0.5

default_rows = df[
    df["threshold"] == default_threshold
]

print("\n========== DEFAULT VS FINAL ==========")

if not default_rows.empty:

    default_row = default_rows.iloc[0]

    print("\nDefault Threshold (0.5)")

    print(
        f"Precision       : "
        f"{default_row['precision']:.6f}"
    )

    print(
        f"Recall          : "
        f"{default_row['recall']:.6f}"
    )

    print(
        f"F1-score        : "
        f"{default_row['f1_score']:.6f}"
    )

    print(
        f"False Positives : "
        f"{int(default_row['false_positives'])}"
    )

    print(
        f"False Negatives : "
        f"{int(default_row['false_negatives'])}"
    )

print("\nFinal Operating Threshold")

print(
    f"Threshold       : "
    f"{final_row['threshold']:.1f}"
)

print(
    f"Precision       : "
    f"{final_row['precision']:.6f}"
)

print(
    f"Recall          : "
    f"{final_row['recall']:.6f}"
)

print(
    f"F1-score        : "
    f"{final_row['f1_score']:.6f}"
)

print(
    f"False Positives : "
    f"{int(final_row['false_positives'])}"
)

print(
    f"False Negatives : "
    f"{int(final_row['false_negatives'])}"
)


# ============================================================
# CREATE FINAL RESULT DATASET
# ============================================================

result = pd.DataFrame([
    {
        "model": "Leakage-Safe Random Forest",

        "validation_type": "Time-Aware",

        "features": 9,

        "threshold": final_row["threshold"],

        "precision": final_row["precision"],

        "recall": final_row["recall"],

        "f1_score": final_row["f1_score"],

        "true_negatives": final_row["true_negatives"],

        "false_positives": final_row["false_positives"],

        "false_negatives": final_row["false_negatives"],

        "true_positives": final_row["true_positives"],

        "minimum_required_recall": MIN_RECALL,

        "maximum_allowed_false_positives": MAX_FALSE_POSITIVES,

        "selection_method": selection_reason
    }
])


# ============================================================
# SAVE FINAL SELECTION
# ============================================================

result.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n========== RESULTS SAVED ==========")

print(
    f"Final threshold selection saved to:"
)

print(
    OUTPUT_PATH
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("LEAKAGE-SAFE THRESHOLD SELECTION COMPLETE")
print("========================================")

print("\nModel:")
print("Leakage-Safe Random Forest")

print("\nValidation:")
print("Time-aware validation")

print("\nFeatures:")
print("9 leakage-safe features")

print(
    f"\nFinal operating threshold: "
    f"{final_row['threshold']:.1f}"
)

print(
    f"Precision: "
    f"{final_row['precision']:.6f}"
)

print(
    f"Recall: "
    f"{final_row['recall']:.6f}"
)

print(
    f"F1-score: "
    f"{final_row['f1_score']:.6f}"
)

print(
    f"False Positives: "
    f"{int(final_row['false_positives'])}"
)

print(
    f"False Negatives: "
    f"{int(final_row['false_negatives'])}"
)

print("\nSelection strategy:")
print(selection_reason)

print("\nNext step:")
print("Finalize the leakage-safe Random Forest model configuration.")