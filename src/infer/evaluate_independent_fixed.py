# src/evaluate_independent_fixed.py

import os
import sys
import contextlib

PROJECT_DIR = "/scratch/project_2010751/TB_Classifier"
os.environ['HF_HOME'] = f"{PROJECT_DIR}/.cache/huggingface"
os.environ['TORCH_HOME'] = f"{PROJECT_DIR}/.cache/torch"

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

from model import MedicalTBClassifier
from dataset_fixed import TBDatasetRobust, get_valid_transforms

def adapt_batch_norm(model, loader, device):
    """
    AdaBN: Update BatchNorm statistics on target domain
    WITHOUT changing model weights
    """
    print("\n🔄 Adapting BatchNorm statistics to target domain...")
    
    was_training = model.training
    model.train()  # Enable BN updates
    
    # Freeze all parameters (only BN running stats will update)
    for param in model.parameters():
        param.requires_grad_(False)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="AdaBN"):
            images = batch['image'].to(device, non_blocking=True)
            if torch.cuda.is_available():
                images = images.to(memory_format=torch.channels_last)
            
            # Forward pass updates BN running mean/var
            _ = model(images)
    
    model.eval()
    
    # Restore gradient tracking
    for param in model.parameters():
        param.requires_grad_(True)
    
    print("✓ BatchNorm adapted to target distribution")

def evaluate_with_tta(model, images, device, temperature=1.0):
    """Test-Time Augmentation: Average original + horizontal flip"""
    
    with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
        # Original
        logits = model(images)
        probs = torch.sigmoid(logits / temperature)
        
        # Horizontal flip
        images_flipped = torch.flip(images, dims=[-1])
        logits_flipped = model(images_flipped)
        probs_flipped = torch.sigmoid(logits_flipped / temperature)
        
        # Average
        probs_avg = (probs + probs_flipped) * 0.5
    
    return probs_avg.cpu().numpy().ravel()

def evaluate_independent_fixed(
    model_path='outputs/tb_sota_final/best_model.pth',
    independent_csv='metadata/independent_test.csv',
    output_dir='outputs/independent_fixed'
):
    """
    Re-evaluate with fixes:
    1. Robust 8/16-bit loading
    2. AdaBN
    3. TTA
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()
    
    print("\n" + "="*60)
    print("🔧 INDEPENDENT EVALUATION - WITH FIXES")
    print("="*60)
    print(f"Device: {device}\n")
    
    # Load model
    print("📥 Loading model...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = MedicalTBClassifier(
        model_name='convnextv2_base.fcmae_ft_in22k_in1k',
        pretrained=False
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    
    # Apply EMA
    ema_shadow = checkpoint.get('ema_shadow', None)
    if isinstance(ema_shadow, dict) and len(ema_shadow) > 0:
        state_dict = model.state_dict()
        for name, shadow_param in ema_shadow.items():
            if name in state_dict and state_dict[name].shape == shadow_param.shape:
                state_dict[name] = shadow_param
        model.load_state_dict(state_dict, strict=False)
        print("✓ EMA weights applied")
    
    if use_cuda:
        model = model.to(memory_format=torch.channels_last)
    
    val_threshold = checkpoint['metrics']['threshold']
    
    # Load data with ROBUST loader
    print(f"\n📂 Loading data with robust preprocessing...")
    
    independent_dataset = TBDatasetRobust(  # Using fixed loader
        csv_file=independent_csv,
        transform=get_valid_transforms(512),
        grayscale=True
    )
    
    independent_loader = torch.utils.data.DataLoader(
        independent_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,  # Reduced as suggested
        pin_memory=use_cuda,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    print(f"✓ Independent dataset: {len(independent_dataset)} images")
    
    # APPLY AdaBN
    adapt_batch_norm(model, independent_loader, device)
    
    model.eval()
    
    # Inference with TTA
    print(f"\n🔍 Running inference with TTA...")
    
    all_preds = []
    all_labels = []
    all_patient_ids = []
    
    with torch.no_grad():
        for batch in tqdm(independent_loader, desc="TTA Inference"):
            images = batch['image'].to(device, non_blocking=use_cuda)
            
            if use_cuda:
                images = images.to(memory_format=torch.channels_last)
            
            # TTA: Average original + flip
            probs = evaluate_with_tta(model, images, device, temperature=1.0)
            
            all_preds.extend(probs)
            all_labels.extend(batch['label'].numpy())
            all_patient_ids.extend(batch['patient_id'])
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    print("\n" + "="*60)
    print("📊 RESULTS WITH FIXES")
    print("="*60)
    
    auc = roc_auc_score(all_labels, all_preds)
    
    binary_preds = (all_preds >= val_threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(all_labels, binary_preds, labels=[0,1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(all_labels, binary_preds)
    
    print(f"\nAfter Fixes:")
    print(f"  ROC-AUC:      {auc:.4f}")
    print(f"  Sensitivity:  {sensitivity:.4f} ({tp}/{tp+fn})")
    print(f"  Specificity:  {specificity:.4f} ({tn}/{tn+fp})")
    print(f"  F1 Score:     {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn:4d}  |  FP: {fp:4d}")
    print(f"  FN: {fn:4d}  |  TP: {tp:4d}")
    
    # Compare to previous
    prev_df = pd.read_csv('outputs/independent_evaluation/independent_predictions.csv')
    prev_auc = roc_auc_score(prev_df['true_label'], prev_df['predicted_prob'])
    
    print(f"\n📈 Improvement:")
    print(f"  Previous AUC: {prev_auc:.4f}")
    print(f"  New AUC:      {auc:.4f}")
    print(f"  Gain:         {auc - prev_auc:+.4f}")
    
    # Save results
    results_df = pd.DataFrame({
        'patient_id': all_patient_ids,
        'true_label': all_labels,
        'predicted_prob': all_preds,
        'predicted_label': binary_preds
    })
    results_df.to_csv(f'{output_dir}/predictions_fixed.csv', index=False)
    
    print(f"\n✓ Results saved to {output_dir}/")
    print("="*60 + "\n")
    
    return auc

if __name__ == "__main__":
    auc_fixed = evaluate_independent_fixed()
