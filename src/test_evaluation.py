# src/test_evaluation.py

import os
import sys
import contextlib

# Set cache FIRST
PROJECT_DIR = "/scratch/project_2010751/TB_Classifier"
os.environ['HF_HOME'] = f"{PROJECT_DIR}/.cache/huggingface"
os.environ['TORCH_HOME'] = f"{PROJECT_DIR}/.cache/torch"
os.environ['TRANSFORMERS_CACHE'] = f"{PROJECT_DIR}/.cache/huggingface"

for cache_dir in [os.environ['HF_HOME'], os.environ['TORCH_HOME']]:
    os.makedirs(cache_dir, exist_ok=True)

print(f"✓ Cache set to: {os.environ['HF_HOME']}")

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    brier_score_loss, f1_score, roc_curve,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from model import MedicalTBClassifier
from createdataset import TBDataset, get_valid_transforms

def evaluate_test_set(model_path, test_csv, output_dir='outputs/test_evaluation'):
    """
    Comprehensive test set evaluation with EMA weights and temperature scaling
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()
    print(f"Device: {device}")
    
    # ============================================================
    # LOAD MODEL WITH EMA WEIGHTS
    # ============================================================
    print("\n📥 Loading best model...")
    
    # FIX: Use weights_only=False for PyTorch 2.6+
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = MedicalTBClassifier(
        model_name='convnextv2_base.fcmae_ft_in22k_in1k',
        pretrained=False
    ).to(device)
    
    # Load base weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    
    # Apply EMA shadow weights if available (better performance)
    ema_shadow = checkpoint.get('ema_shadow', None)
    if isinstance(ema_shadow, dict) and len(ema_shadow) > 0:
        state_dict = model.state_dict()
        ema_applied = 0
        for name, shadow_param in ema_shadow.items():
            if name in state_dict and state_dict[name].shape == shadow_param.shape:
                state_dict[name] = shadow_param
                ema_applied += 1
        if ema_applied > 0:
            model.load_state_dict(state_dict, strict=False)
            print(f"✓ Applied EMA weights ({ema_applied} parameters)")
    else:
        print("ℹ️  No EMA weights found, using standard weights")
    
    # Optimize for inference
    if use_cuda:
        model = model.to(memory_format=torch.channels_last)
        print("✓ Using channels_last memory format")
    
    model.eval()
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']+1}")
    print(f"  Validation AUC: {checkpoint['metrics']['auc']:.4f}")
    
    # ============================================================
    # LOAD TEMPERATURE SCALING (if available)
    # ============================================================
    temperature = 1.0
    temp_path = os.path.join(os.path.dirname(model_path), "temperature.pth")
    
    if os.path.exists(temp_path):
        temp_tensor = torch.load(temp_path, map_location=device, weights_only=False)
        temperature = temp_tensor.item() if torch.is_tensor(temp_tensor) else float(temp_tensor)
        print(f"✓ Using calibrated temperature T={temperature:.4f}")
    else:
        print("ℹ️  No temperature scaling found (using T=1.0)")
    
    # ============================================================
    # LOAD TEST DATA
    # ============================================================
    test_dataset = TBDataset(
        csv_file=test_csv,
        transform=get_valid_transforms(512),
        grayscale=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=use_cuda
    )
    
    print(f"✓ Test set: {len(test_dataset)} images")
    
    # ============================================================
    # INFERENCE
    # ============================================================
    print("\n🔍 Running inference on test set...")
    
    autocast = torch.cuda.amp.autocast if use_cuda else contextlib.nullcontext
    
    all_logits = []
    all_preds = []
    all_labels = []
    all_patient_ids = []
    all_datasets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device, non_blocking=use_cuda)
            
            if use_cuda:
                images = images.to(memory_format=torch.channels_last)
            
            with autocast():
                logits = model(images)
            
            # Apply temperature scaling
            scaled_logits = logits / temperature
            probs = torch.sigmoid(scaled_logits).cpu().numpy().ravel()
            
            all_logits.extend(logits.cpu().numpy().ravel())
            all_preds.extend(probs)
            all_labels.extend(batch['label'].numpy())
            all_patient_ids.extend(batch['patient_id'])
            all_datasets.extend(batch['dataset'])
    
    all_logits = np.array(all_logits)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # ============================================================
    # CALCULATE METRICS
    # ============================================================
    print("\n" + "="*60)
    print("📊 TEST SET RESULTS")
    print("="*60)
    
    # Overall metrics
    auc = roc_auc_score(all_labels, all_preds)
    prauc = average_precision_score(all_labels, all_preds)
    brier = brier_score_loss(all_labels, all_preds)
    
    print(f"\nOverall Performance:")
    print(f"  ROC-AUC:          {auc:.4f}")
    print(f"  PR-AUC:           {prauc:.4f}")
    print(f"  Brier Score:      {brier:.4f}")
    
    # Find threshold for 95% specificity (from test data for reporting)
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    
    # Threshold at 95% specificity
    spec_95_idx = np.where(fpr <= 0.05)[0]
    if len(spec_95_idx) > 0:
        idx_95 = spec_95_idx[-1]  # Last index where FPR <= 0.05
        thresh_95spec = thresholds[idx_95]
        sens_at_95spec = tpr[idx_95]
    else:
        thresh_95spec = 0.5
        sens_at_95spec = 0.0
    
    # Use validation-derived threshold for classification
    val_threshold = checkpoint['metrics']['threshold']
    
    print(f"\nThresholds:")
    print(f"  Val-derived (F1-optimal):     {val_threshold:.4f}")
    print(f"  Test-derived (@95% Spec):     {thresh_95spec:.4f}")
    print(f"  Sensitivity @ 95% Spec:       {sens_at_95spec:.4f}")
    
    # Classification metrics using validation threshold
    binary_preds = (all_preds >= val_threshold).astype(int)
    f1 = f1_score(all_labels, binary_preds)
    
    tn, fp, fn, tp = confusion_matrix(all_labels, binary_preds, labels=[0,1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    print(f"\nClassification Metrics (threshold={val_threshold:.4f}):")
    print(f"  Accuracy:         {accuracy:.4f}")
    print(f"  Sensitivity:      {sensitivity:.4f} ({tp}/{tp+fn})")
    print(f"  Specificity:      {specificity:.4f} ({tn}/{tn+fp})")
    print(f"  PPV (Precision):  {ppv:.4f}")
    print(f"  NPV:              {npv:.4f}")
    print(f"  F1 Score:         {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negative:  {tn:4d}  |  False Positive: {fp:4d}")
    print(f"  False Negative: {fn:4d}  |  True Positive:  {tp:4d}")
    
    # ============================================================
    # PER-DATASET BREAKDOWN
    # ============================================================
    print(f"\n" + "="*60)
    print("Per-Dataset Performance:")
    print("="*60)
    
    for dataset_name in sorted(set(all_datasets)):
        mask = np.array(all_datasets) == dataset_name
        n_samples = mask.sum()
        
        if n_samples > 0:
            dataset_auc = roc_auc_score(all_labels[mask], all_preds[mask])
            dataset_prauc = average_precision_score(all_labels[mask], all_preds[mask])
            dataset_brier = brier_score_loss(all_labels[mask], all_preds[mask])
            
            print(f"\n{dataset_name}:")
            print(f"  Samples:    {n_samples:4d}")
            print(f"  ROC-AUC:    {dataset_auc:.4f}")
            print(f"  PR-AUC:     {dataset_prauc:.4f}")
            print(f"  Brier:      {dataset_brier:.4f}")
    
    # ============================================================
    # SAVE RESULTS
    # ============================================================
    results_df = pd.DataFrame({
        'patient_id': all_patient_ids,
        'dataset': all_datasets,
        'true_label': all_labels,
        'logit': all_logits,
        'predicted_prob': all_preds,
        'predicted_label': binary_preds
    })
    results_df.to_csv(f'{output_dir}/test_predictions.csv', index=False)
    print(f"\n✓ Predictions saved to {output_dir}/test_predictions.csv")
    
    # ============================================================
    # PLOTS
    # ============================================================
    
    # ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, linewidth=2, label=f'Model (AUC = {auc:.4f})', color='#2E86AB')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    # Mark 95% specificity point
    plt.plot(0.05, sens_at_95spec, 'ro', markersize=10, 
             label=f'95% Spec (Sens={sens_at_95spec:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curve - Test Set\nTB Classifier Performance', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curve_test.png', dpi=300, bbox_inches='tight')
    print(f"✓ ROC curve saved")
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        [[tn, fp], [fn, tp]], 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Predicted Negative', 'Predicted Positive'],
        yticklabels=['True Negative', 'True Positive'],
        cbar_kws={'label': 'Count'}
    )
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(f'Confusion Matrix - Test Set\n(Threshold={val_threshold:.4f})', 
              fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix_test.png', dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved")
    
    # Prediction distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # By true label
    axes[0].hist(all_preds[all_labels==0], bins=50, alpha=0.6, label='True Negative', color='blue')
    axes[0].hist(all_preds[all_labels==1], bins=50, alpha=0.6, label='True Positive', color='red')
    axes[0].axvline(val_threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold={val_threshold:.3f}')
    axes[0].set_xlabel('Predicted Probability', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('Prediction Distribution by True Label', fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Calibration-like view
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_counts = []
    bin_accuracies = []
    
    for i in range(len(bins)-1):
        mask = (all_preds >= bins[i]) & (all_preds < bins[i+1])
        if mask.sum() > 0:
            bin_counts.append(mask.sum())
            bin_accuracies.append(all_labels[mask].mean())
        else:
            bin_counts.append(0)
            bin_accuracies.append(0)
    
    axes[1].bar(bin_centers, bin_accuracies, width=0.08, alpha=0.6, label='Actual')
    axes[1].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    axes[1].set_xlabel('Predicted Probability Bin', fontsize=11)
    axes[1].set_ylabel('Fraction of Positives', fontsize=11)
    axes[1].set_title(f'Calibration Plot (Brier={brier:.4f})', fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/prediction_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Prediction analysis saved")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*60)
    print("✅ TEST EVALUATION COMPLETE")
    print("="*60)
    print(f"\nKey Results:")
    print(f"  • ROC-AUC:    {auc:.4f}")
    print(f"  • Sensitivity: {sensitivity:.4f}")
    print(f"  • Specificity: {specificity:.4f}")
    print(f"  • Brier Score: {brier:.4f}")
    print(f"\nOutputs saved to: {output_dir}/")
    print("="*60 + "\n")
    
    return {
        'auc': auc,
        'prauc': prauc,
        'brier': brier,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1,
        'sens_at_95spec': sens_at_95spec
    }

if __name__ == "__main__":
    results = evaluate_test_set(
        model_path='outputs/tb_sota_final/best_model.pth',
        test_csv='metadata/test.csv',
        output_dir='outputs/test_evaluation'
    )
