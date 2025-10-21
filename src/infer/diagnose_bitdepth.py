# src/diagnose_bitdepth.py

import cv2
import numpy as np
import glob
import random
from pathlib import Path

def diagnose_image_properties():
    """Check if DA+DB images have different bit depth or scaling"""
    
    print("="*60)
    print("🔍 IMAGE PROPERTIES DIAGNOSTIC")
    print("="*60)
    
    # Check training data samples
    print("\n1. TRAINING DATA (TBX11K sick):")
    print("-"*60)
    train_paths = list(Path('tb_datasets/tbx11k/TBX11K/imgs/sick').glob('*.png'))[:5]
    
    for p in train_paths:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        img_gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        print(f"{p.name:30} dtype={img.dtype}, shape={img.shape}, "
              f"range=[{img.min()}-{img.max()}], gray={img_gray.dtype}")
    
    # Check DA+DB independent data
    print("\n2. INDEPENDENT DATA (DA+DB):")
    print("-"*60)
    
    dadb_paths = (list(Path('tb_datasets/tbx11k/TBX11K/imgs/extra/da+db/train').glob('*.png'))[:5] +
                  list(Path('tb_datasets/tbx11k/TBX11K/imgs/extra/da+db/train').glob('*.jpg'))[:5])
    
    for p in dadb_paths:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        img_gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        print(f"{p.name:30} dtype={img.dtype}, shape={img.shape}, "
              f"range=[{img.min()}-{img.max()}], gray={img_gray.dtype}")
    
    # Check test set predictions
    print("\n3. PREDICTION STATISTICS:")
    print("-"*60)
    
    import pandas as pd
    
    test_df = pd.read_csv('outputs/test_evaluation/test_predictions.csv')
    ind_df = pd.read_csv('outputs/independent_evaluation/independent_predictions.csv')
    
    print(f"\nTest Set (working):")
    print(f"  TB cases - Mean prob: {test_df[test_df['true_label']==1]['predicted_prob'].mean():.4f}")
    print(f"  TB cases - Median:    {test_df[test_df['true_label']==1]['predicted_prob'].median():.4f}")
    print(f"  TB cases - Range:     [{test_df[test_df['true_label']==1]['predicted_prob'].min():.4f}, "
          f"{test_df[test_df['true_label']==1]['predicted_prob'].max():.4f}]")
    
    print(f"\nIndependent (failing):")
    print(f"  TB cases - Mean prob: {ind_df[ind_df['true_label']==1]['predicted_prob'].mean():.4f}")
    print(f"  TB cases - Median:    {ind_df[ind_df['true_label']==1]['predicted_prob'].median():.4f}")
    print(f"  TB cases - Range:     [{ind_df[ind_df['true_label']==1]['predicted_prob'].min():.4f}, "
          f"{ind_df[ind_df['true_label']==1]['predicted_prob'].max():.4f}]")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    diagnose_image_properties()
