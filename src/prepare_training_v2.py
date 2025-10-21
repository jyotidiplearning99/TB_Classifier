# src/prepare_training_v2.py

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def create_diverse_training_data():
    """
    Create new training dataset including DA+DB for robust generalization
    
    Strategy:
    - Keep original train/val/test split intact
    - Add DA+DB split: 70% train, 20% val, 10% test
    - Results in ~10% DA+DB representation across all splits
    """
    
    print("="*70)
    print("📦 CREATING DIVERSE TRAINING DATASET V2 (WITH DA+DB)")
    print("="*70)
    
    # Load existing data
    train_df = pd.read_csv('metadata/train.csv')
    val_df = pd.read_csv('metadata/val.csv')
    test_df = pd.read_csv('metadata/test.csv')
    dadb_df = pd.read_csv('metadata/independent_test.csv')
    
    print(f"\n📊 Original Dataset Sizes:")
    print(f"{'Dataset':<15} {'Images':<8} {'TB Pos':<8} {'TB Neg':<8} {'% Positive'}")
    print("-"*70)
    
    for name, df in [('Train', train_df), ('Validation', val_df), ('Test', test_df), ('DA+DB', dadb_df)]:
        n_total = len(df)
        n_pos = (df['label'] == 1).sum()
        n_neg = (df['label'] == 0).sum()
        pct_pos = n_pos / n_total * 100
        print(f"{name:<15} {n_total:<8} {n_pos:<8} {n_neg:<8} {pct_pos:.1f}%")
    
    # Split DA+DB: 70% train, 20% val, 10% test
    print(f"\n🔪 Splitting DA+DB:")
    print(f"   Strategy: 70% train, 20% val, 10% test")
    print(f"   Stratified by label to maintain class balance")
    
    # First split: 70% train, 30% temp
    dadb_train, dadb_temp = train_test_split(
        dadb_df,
        test_size=0.3,
        stratify=dadb_df['label'],
        random_state=42
    )
    
    # Second split: from 30%, take 2/3 for val (20% of total), 1/3 for test (10% of total)
    dadb_val, dadb_test = train_test_split(
        dadb_temp,
        test_size=0.33,  # 1/3 of 30% = 10% of total
        stratify=dadb_temp['label'],
        random_state=42
    )
    
    print(f"\n   DA+DB Split Results:")
    print(f"   {'Split':<15} {'Images':<8} {'TB Pos':<8} {'TB Neg':<8} {'% of DA+DB'}")
    print(f"   {'-'*60}")
    
    for name, df in [('Train', dadb_train), ('Val', dadb_val), ('Test', dadb_test)]:
        n_total = len(df)
        n_pos = (df['label'] == 1).sum()
        n_neg = (df['label'] == 0).sum()
        pct = n_total / len(dadb_df) * 100
        print(f"   {name:<15} {n_total:<8} {n_pos:<8} {n_neg:<8} {pct:.1f}%")
    
    # Combine with original data
    new_train = pd.concat([train_df, dadb_train], ignore_index=True)
    new_val = pd.concat([val_df, dadb_val], ignore_index=True)
    new_test = pd.concat([test_df, dadb_test], ignore_index=True)
    
    # Save new splits
    output_dir = Path('metadata')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    new_train.to_csv('metadata/train_v2.csv', index=False)
    new_val.to_csv('metadata/val_v2.csv', index=False)
    new_test.to_csv('metadata/test_v2.csv', index=False)
    
    print(f"\n{'='*70}")
    print("✅ NEW DIVERSE DATASETS CREATED")
    print(f"{'='*70}")
    
    print(f"\n📈 New Dataset Sizes:")
    print(f"{'Dataset':<15} {'Total':<8} {'Original':<10} {'DA+DB':<8} {'% DA+DB'}")
    print("-"*70)
    
    for name, new_df, old_df, dadb_subset in [
        ('Train V2', new_train, train_df, dadb_train),
        ('Val V2', new_val, val_df, dadb_val),
        ('Test V2', new_test, test_df, dadb_test)
    ]:
        n_total = len(new_df)
        n_original = len(old_df)
        n_dadb = len(dadb_subset)
        pct_dadb = n_dadb / n_total * 100
        print(f"{name:<15} {n_total:<8} {n_original:<10} {n_dadb:<8} {pct_dadb:.1f}%")
    
    # Class distribution in new training set
    print(f"\n📊 Class Distribution in Train V2:")
    print(f"   TB Negative: {(new_train['label']==0).sum():5d} ({(new_train['label']==0).sum()/len(new_train)*100:.1f}%)")
    print(f"   TB Positive: {(new_train['label']==1).sum():5d} ({(new_train['label']==1).sum()/len(new_train)*100:.1f}%)")
    
    # Dataset diversity in new training set
    print(f"\n🌍 Dataset Diversity in Train V2:")
    dataset_counts = new_train['dataset'].value_counts()
    for dataset, count in dataset_counts.items():
        pct = count / len(new_train) * 100
        print(f"   {dataset:<30} {count:5d} ({pct:5.1f}%)")
    
    print(f"\n{'='*70}")
    print("📝 FILES CREATED:")
    print(f"{'='*70}")
    print(f"   ✓ metadata/train_v2.csv  ({len(new_train):,} images)")
    print(f"   ✓ metadata/val_v2.csv    ({len(new_val):,} images)")
    print(f"   ✓ metadata/test_v2.csv   ({len(new_test):,} images)")
    
    print(f"\n{'='*70}")
    print("🚀 NEXT STEPS:")
    print(f"{'='*70}")
    print("   1. Review the splits above")
    print("   2. Run training with: sbatch train_job_v2.sh")
    print("   3. Expected improvement on DA+DB: AUC 0.56 → 0.88-0.92")
    print(f"{'='*70}\n")
    
    return new_train, new_val, new_test

if __name__ == "__main__":
    train_v2, val_v2, test_v2 = create_diverse_training_data()
