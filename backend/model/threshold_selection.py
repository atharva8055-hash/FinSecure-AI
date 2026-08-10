import pandas as pd

# Load threshold analysis results
file_path = "dataset/threshold_analysis.csv"

print("\n========== LOADING THRESHOLD RESULTS ==========")

df = pd.read_csv(file_path)

print("Threshold results loaded successfully.")

print("\n========== THRESHOLD RESULTS ==========")
print(df.to_string(index=False))

# Find best threshold according to F1
best_f1 = df.loc[df["f1_score"].idxmax()]

print("\n========== BEST F1 THRESHOLD ==========")
print("Threshold:", best_f1["threshold"])
print("Precision:", best_f1["precision"])
print("Recall:", best_f1["recall"])
print("F1-score:", best_f1["f1_score"])
print("False Positives:", best_f1["false_positives"])
print("False Negatives:", best_f1["false_negatives"])

# Find threshold with high recall and lower false positives
high_recall = df[df["recall"] >= 0.90]

if not high_recall.empty:

    practical_threshold = high_recall.loc[
        high_recall["false_positives"].idxmin()
    ]

    print("\n========== HIGH-RECALL THRESHOLD ==========")
    print("Threshold:", practical_threshold["threshold"])
    print("Precision:", practical_threshold["precision"])
    print("Recall:", practical_threshold["recall"])
    print("F1-score:", practical_threshold["f1_score"])
    print("False Positives:", practical_threshold["false_positives"])
    print("False Negatives:", practical_threshold["false_negatives"])

else:

    practical_threshold = None

    print("\nNo threshold found with recall >= 90%.")

# Compare default threshold and selected threshold
default_threshold = df[df["threshold"] == 0.5].iloc[0]

print("\n========== DEFAULT VS SELECTED ==========")

print("\nDefault Threshold (0.5)")
print("Precision:", default_threshold["precision"])
print("Recall:", default_threshold["recall"])
print("F1-score:", default_threshold["f1_score"])
print("False Positives:", default_threshold["false_positives"])
print("False Negatives:", default_threshold["false_negatives"])

print("\nBest F1 Threshold")
print("Threshold:", best_f1["threshold"])
print("Precision:", best_f1["precision"])
print("Recall:", best_f1["recall"])
print("F1-score:", best_f1["f1_score"])
print("False Positives:", best_f1["false_positives"])
print("False Negatives:", best_f1["false_negatives"])

# Save recommendation
recommendation = pd.DataFrame({
    "selection_type": [
        "best_f1",
        "high_recall"
    ],
    "threshold": [
        best_f1["threshold"],
        practical_threshold["threshold"] if practical_threshold is not None else None
    ],
    "precision": [
        best_f1["precision"],
        practical_threshold["precision"] if practical_threshold is not None else None
    ],
    "recall": [
        best_f1["recall"],
        practical_threshold["recall"] if practical_threshold is not None else None
    ],
    "f1_score": [
        best_f1["f1_score"],
        practical_threshold["f1_score"] if practical_threshold is not None else None
    ],
    "false_positives": [
        best_f1["false_positives"],
        practical_threshold["false_positives"] if practical_threshold is not None else None
    ],
    "false_negatives": [
        best_f1["false_negatives"],
        practical_threshold["false_negatives"] if practical_threshold is not None else None
    ]
})

output_path = "dataset/threshold_selection.csv"

recommendation.to_csv(output_path, index=False)

print("\n========== THRESHOLD SELECTION COMPLETE ==========")
print("Results saved to:")
print(output_path)

print("\nNext step:")
print("Proceed to improved fraud detection models.")