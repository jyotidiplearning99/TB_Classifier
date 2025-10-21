import pandas as pd
import matplotlib.pyplot as plt
import cv2

df = pd.read_csv('outputs/test_evaluation/test_predictions.csv')

# False negatives (missed TB cases)
fn = df[(df['true_label'] == 1) & (df['predicted_label'] == 0)].copy()
fn = fn.sort_values('predicted_prob')

print(f"\n{'='*60}")
print(f"FALSE NEGATIVE ANALYSIS (n={len(fn)})")
print(f"{'='*60}\n")

print("Most confident misses (lowest predicted probability):")
print(fn[['patient_id', 'dataset', 'predicted_prob']].head(5))

print("\nLeast confident misses (closest to threshold):")
print(fn[['patient_id', 'dataset', 'predicted_prob']].tail(5))

print(f"\nFalse negative rate by dataset:")
for dataset in fn['dataset'].unique():
    n_fn = (fn['dataset'] == dataset).sum()
    n_tp = ((df['dataset'] == dataset) & (df['true_label'] == 1) & (df['predicted_label'] == 1)).sum()
    total = n_fn + n_tp
    rate = n_fn / total * 100 if total > 0 else 0
    print(f"  {dataset:20} {n_fn}/{total} ({rate:.1f}%)")

# False positives
fp = df[(df['true_label'] == 0) & (df['predicted_label'] == 1)].copy()
fp = fp.sort_values('predicted_prob', ascending=False)

print(f"\n{'='*60}")
print(f"FALSE POSITIVE ANALYSIS (n={len(fp)})")
print(f"{'='*60}\n")

print("Most confident errors (highest predicted probability):")
print(fp[['patient_id', 'dataset', 'predicted_prob']].head(5))

