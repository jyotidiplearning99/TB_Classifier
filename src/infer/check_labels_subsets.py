# src/check_labels_subsets.py

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

print("="*60)
print("🔍 LABEL & SUBSET SANITY CHECK")
print("="*60)

# Check label distribution
print("\n1. Label Distribution Check:")
df = pd.read_csv('metadata/independent_test.csv')
print(df.groupby(['dataset', 'label']).size().unstack(fill_value=0))

print("\n2. Sample Patient IDs (check naming pattern):")
print("\nFirst 10:")
print(df['patient_id'].head(10).tolist())
print("\nLast 10:")
print(df['patient_id'].tail(10).tolist())

# Check predictions by subset
print("\n" + "="*60)
print("3. AUC by Image Format:")
print("="*60)

pred_df = pd.read_csv('outputs/independent_fixed/predictions_fixed.csv')
merged = pred_df.merge(df[['patient_id', 'dataset']], on='patient_id')

# Extract file extension from patient_id
merged['ext'] = merged['patient_id'].str.extract(r'\.([a-z]+)$')[0].fillna('png')

print("\nBy extension:")
for ext in merged['ext'].unique():
    subset = merged[merged['ext'] == ext]
    if subset['true_label'].nunique() == 2:
        auc = roc_auc_score(subset['true_label'], subset['predicted_prob'])
        n_pos = (subset['true_label'] == 1).sum()
        n_neg = (subset['true_label'] == 0).sum()
        print(f"  {ext:5s}: AUC={auc:.4f}  (n={len(subset)}, pos={n_pos}, neg={n_neg})")
    else:
        print(f"  {ext:5s}: Cannot compute (only one class present)")

# Check by DA vs DB (if distinguishable from naming)
print("\n" + "="*60)
print("4. AUC by Dataset Origin:")
print("="*60)

# DA files typically have no 'x' in suffix, DB have 'x'
merged['origin'] = merged['patient_id'].str.contains('nx|px').map({True: 'DB', False: 'DA'})

print("\nBy origin (DA vs DB):")
for origin in ['DA', 'DB']:
    subset = merged[merged['origin'] == origin]
    if len(subset) > 0 and subset['true_label'].nunique() == 2:
        auc = roc_auc_score(subset['true_label'], subset['predicted_prob'])
        n_pos = (subset['true_label'] == 1).sum()
        n_neg = (subset['true_label'] == 0).sum()
        mean_prob = subset['predicted_prob'].mean()
        print(f"  {origin:5s}: AUC={auc:.4f}  Mean_prob={mean_prob:.4f}  (n={len(subset)}, pos={n_pos}, neg={n_neg})")

# Check TB positive cases specifically
print("\n" + "="*60)
print("5. TB Positive Cases Analysis:")
print("="*60)

tb_cases = merged[merged['true_label'] == 1]
print(f"\nTotal TB cases: {len(tb_cases)}")
print(f"Mean prediction: {tb_cases['predicted_prob'].mean():.4f}")
print(f"Median prediction: {tb_cases['predicted_prob'].median():.4f}")
print(f"Min prediction: {tb_cases['predicted_prob'].min():.4f}")
print(f"Max prediction: {tb_cases['predicted_prob'].max():.4f}")

print("\nHighest predicted TB cases (correct):")
print(tb_cases.nlargest(5, 'predicted_prob')[['patient_id', 'predicted_prob']])

print("\nLowest predicted TB cases (model missed):")
print(tb_cases.nsmallest(5, 'predicted_prob')[['patient_id', 'predicted_prob']])

print("\n" + "="*60)
