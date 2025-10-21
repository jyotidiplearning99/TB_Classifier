# src/analyze_dadb_predictions.py

import pandas as pd
import numpy as np

# Load predictions
df = pd.read_csv('outputs/tb_v2_robust/dadb_holdout_predictions.csv')

print("="*70)
print("🔍 DA+DB PREDICTION ANALYSIS")
print("="*70)

print(f"\nDataset size: {len(df)} images")
print(f"  TB Positive: {(df['true_label']==1).sum()}")
print(f"  TB Negative: {(df['true_label']==0).sum()}")

print(f"\nValidation threshold used: 0.5874")

# TB Cases analysis
n_tb = (df['true_label']==1).sum()
print(f"\n📊 TB Cases (n={n_tb}):")
tb_cases = df[df['true_label']==1]['predicted_prob'].values
print(f"  Mean prob:   {tb_cases.mean():.4f}")
print(f"  Median prob: {np.median(tb_cases):.4f}")
print(f"  Min prob:    {tb_cases.min():.4f}")
print(f"  Max prob:    {tb_cases.max():.4f}")
print(f"\n  All TB predictions (sorted):")
for i, prob in enumerate(sorted(tb_cases, reverse=True), 1):
    print(f"    TB case {i}: {prob:.4f}")

# Healthy cases analysis
n_healthy = (df['true_label']==0).sum()
print(f"\n📊 Healthy Cases (n={n_healthy}):")
healthy = df[df['true_label']==0]['predicted_prob'].values
print(f"  Mean prob:   {healthy.mean():.4f}")
print(f"  Median prob: {np.median(healthy):.4f}")
print(f"  Min prob:    {healthy.min():.4f}")
print(f"  Max prob:    {healthy.max():.4f}")

print(f"\n🎯 Analysis:")
print(f"  Threshold: 0.5874")
print(f"  TB cases above threshold: {(tb_cases >= 0.5874).sum()}/7")
print(f"  Healthy above threshold:  {(healthy >= 0.5874).sum()}/11")
print(f"  Separation gap: {tb_cases.max():.4f} (max TB) vs {healthy.min():.4f} (min healthy)")

# Try different thresholds
print(f"\n📈 Performance at Different Thresholds:")
print(f"{'Threshold':<12} {'Sensitivity':<15} {'Specificity':<15} {'F1 Score'}")
print("-"*60)

from sklearn.metrics import f1_score as compute_f1

for thresh in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.5874]:
    preds = (df['predicted_prob'] >= thresh).astype(int)
    tp = ((df['true_label']==1) & (preds==1)).sum()
    fn = ((df['true_label']==1) & (preds==0)).sum()
    tn = ((df['true_label']==0) & (preds==0)).sum()
    fp = ((df['true_label']==0) & (preds==1)).sum()
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = compute_f1(df['true_label'], preds)
    
    marker = " ← BEST" if f1 > 0.5 else ""
    print(f"{thresh:<12.4f} {sens*100:<14.1f}% {spec*100:<14.1f}% {f1:.4f}{marker}")

# Recommendation
print("\n" + "="*70)
print("💡 RECOMMENDATION")
print("="*70)

if tb_cases.max() < 0.2:
    print("\n❌ TB predictions are VERY LOW (max < 0.2)")
    print("   → Model still struggling with DA+DB domain")
    print("   → Need MORE DA+DB data in training (currently only 0.96%)")
    print("   → Or try domain-specific fine-tuning")
elif tb_cases.max() < 0.4:
    print("\n⚠️  TB predictions are LOW (max < 0.4)")
    print("   → Threshold calibration issue")
    print("   → Use threshold 0.1-0.2 for DA+DB")
    print("   → Consider more DA+DB representation in training")
else:
    print("\n✅ TB predictions are REASONABLE (max >= 0.4)")
    print("   → Simple threshold adjustment should work")
    print(f"   → Optimal threshold appears to be around {tb_cases.mean():.3f}")

print("\n" + "="*70)
