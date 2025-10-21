# src/evaluate_independent.py

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

def evaluate_independent_dataset(
    model_path='outputs/tb_sota_final/best_model.pth',
    independent_csv='metadata/independent_test.csv',
    test_csv='metadata/test.csv',
    output_dir='outputs/independent_evaluation'
):
    """
    Evaluate on completely independent dataset and compare to original test set
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()
    
    print("\n" + "="*60)
    print("🌍 INDEPENDENT DATASET EVALUATION")
    print("="*60)
    print(f"Device: {device}\n")
    
    # ============================================================
    # LOAD MODEL WITH EMA
    # ============================================================
    print("📥 Loading trained model...")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = MedicalTBClassifier(
        model_name='convnextv2_base.fcmae_ft_in22k_in1k',
        pretrained=False
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    
    # Apply EMA weights
    ema_shadow = checkpoint.get('ema_shadow', None)
    if isinstance(ema_shadow, dict) and len(ema_shadow) > 0:
        state_dict = model.state_dict()
        for name, shadow_param in ema_shadow.items():
            if name in state_dict and state_dict[name].shape == shadow_param.shape:
                state_dict[name] = shadow_param
        model.load_state_dict(state_dict, strict=False)
        print("✓ Applied EMA weights")
    
    if use_cuda:
        model = model.to(memory_format=torch.channels_last)
    
    model.eval()
    
    print(f"✓ Model from epoch {checkpoint['epoch']+1}")
    print(f"  Original validation AUC: {checkpoint['metrics']['auc']:.4f}")
    
    # Get validation threshold
    val_threshold = checkpoint['metrics']['threshold']
    
    # ============================================================
    # LOAD INDEPENDENT DATASET
    # ============================================================
    print(f"\n📂 Loading independent dataset...")
    
    independent_dataset = TBDataset(
        csv_file=independent_csv,
        transform=get_valid_transforms(512),
        grayscale=True
    )
    
    independent_loader = torch.utils.data.DataLoader(
        independent_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=use_cuda
    )
    
    print(f"✓ Independent dataset: {len(independent_dataset)} images")
    
    # ============================================================
    # INFERENCE ON INDEPENDENT DATA
    # ============================================================
    print(f"\n🔍 Running inference on independent dataset...")
    
    autocast = torch.cuda.amp.autocast if use_cuda else contextlib.nullcontext
    
    ind_preds = []
    ind_labels = []
    ind_patient_ids = []
    
    with torch.no_grad():
        for batch in tqdm(independent_loader, desc="Independent"):
            images = batch['image'].to(device, non_blocking=use_cuda)
            
            if use_cuda:
                images = images.to(memory_format=torch.channels_last)
            
            with autocast():
                logits = model(images)
            
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            
            ind_preds.extend(probs)
            ind_labels.extend(batch['label'].numpy())
            ind_patient_ids.extend(batch['patient_id'])
    
    ind_preds = np.array(ind_preds)
    ind_labels = np.array(ind_labels)
    
    # ============================================================
    # ALSO LOAD ORIGINAL TEST SET RESULTS FOR COMPARISON
    # ============================================================
    print(f"\n📊 Loading original test set results for comparison...")
    
    test_results = pd.read_csv('outputs/test_evaluation/test_predictions.csv')
    test_preds = test_results['predicted_prob'].values
    test_labels = test_results['true_label'].values
    
    # ============================================================
    # CALCULATE METRICS - INDEPENDENT
    # ============================================================
    print("\n" + "="*60)
    print("📊 INDEPENDENT DATASET RESULTS")
    print("="*60)
    
    ind_auc = roc_auc_score(ind_labels, ind_preds)
    ind_prauc = average_precision_score(ind_labels, ind_preds)
    ind_brier = brier_score_loss(ind_labels, ind_preds)
    
    # Use validation-derived threshold
    ind_binary_preds = (ind_preds >= val_threshold).astype(int)
    ind_f1 = f1_score(ind_labels, ind_binary_preds)
    
    tn, fp, fn, tp = confusion_matrix(ind_labels, ind_binary_preds, labels=[0,1]).ravel()
    ind_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    ind_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ind_accuracy = (tp + tn) / (tp + tn + fp + fn)
    ind_ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    ind_npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # Sensitivity at 95% specificity
    fpr, tpr, _ = roc_curve(ind_labels, ind_preds)
    idx = np.searchsorted(fpr, 0.05, side='left')
    idx = min(max(idx, 0), len(tpr) - 1)
    ind_sens_95spec = tpr[idx]
    
    print(f"\nPerformance Metrics:")
    print(f"  ROC-AUC:          {ind_auc:.4f}")
    print(f"  PR-AUC:           {ind_prauc:.4f}")
    print(f"  Brier Score:      {ind_brier:.4f}")
    print(f"  Accuracy:         {ind_accuracy:.4f}")
    print(f"  Sensitivity:      {ind_sensitivity:.4f} ({tp}/{tp+fn})")
    print(f"  Specificity:      {ind_specificity:.4f} ({tn}/{tn+fp})")
    print(f"  F1 Score:         {ind_f1:.4f}")
    print(f"  PPV:              {ind_ppv:.4f}")
    print(f"  NPV:              {ind_npv:.4f}")
    print(f"  Sens@95%Spec:     {ind_sens_95spec:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn:4d}  |  FP: {fp:4d}")
    print(f"  FN: {fn:4d}  |  TP: {tp:4d}")
    
    # ============================================================
    # CALCULATE METRICS - ORIGINAL TEST SET
    # ============================================================
    print("\n" + "="*60)
    print("📊 ORIGINAL TEST SET (for comparison)")
    print("="*60)
    
    test_auc = roc_auc_score(test_labels, test_preds)
    test_binary_preds = (test_preds >= val_threshold).astype(int)
    
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(test_labels, test_binary_preds, labels=[0,1]).ravel()
    test_sensitivity = tp_t / (tp_t + fn_t)
    test_specificity = tn_t / (tn_t + fp_t)
    
    print(f"\nOriginal Test Set:")
    print(f"  ROC-AUC:          {test_auc:.4f}")
    print(f"  Sensitivity:      {test_sensitivity:.4f}")
    print(f"  Specificity:      {test_specificity:.4f}")
    
    # ============================================================
    # COMPARISON TABLE
    # ============================================================
    print("\n" + "="*60)
    print("📊 GENERALIZATION ANALYSIS")
    print("="*60)
    
    comparison = pd.DataFrame({
        'Metric': ['ROC-AUC', 'Sensitivity', 'Specificity', 'F1 Score', 'Brier Score'],
        'Validation': [
            checkpoint['metrics']['auc'],
            checkpoint['metrics']['sensitivity'],
            checkpoint['metrics']['specificity'],
            checkpoint['metrics']['f1'],
            checkpoint['metrics']['brier']
        ],
        'Test Set': [test_auc, test_sensitivity, test_specificity, 
                     f1_score(test_labels, test_binary_preds),
                     brier_score_loss(test_labels, test_preds)],
        'Independent': [ind_auc, ind_sensitivity, ind_specificity, ind_f1, ind_brier]
    })
    
    # Calculate differences
    comparison['Test-Val Diff'] = comparison['Test Set'] - comparison['Validation']
    comparison['Indep-Test Diff'] = comparison['Independent'] - comparison['Test Set']
    
    print("\n" + comparison.to_string(index=False))
    
    # ============================================================
    # SAVE RESULTS
    # ============================================================
    results_df = pd.DataFrame({
        'patient_id': ind_patient_ids,
        'true_label': ind_labels,
        'predicted_prob': ind_preds,
        'predicted_label': ind_binary_preds
    })
    results_df.to_csv(f'{output_dir}/independent_predictions.csv', index=False)
    
    comparison.to_csv(f'{output_dir}/comparison_metrics.csv', index=False)
    
    print(f"\n✓ Results saved to {output_dir}/")
    
    # ============================================================
    # VISUALIZATIONS
    # ============================================================
    
    # ROC curves comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ROC curves
    fpr_test, tpr_test, _ = roc_curve(test_labels, test_preds)
    fpr_ind, tpr_ind, _ = roc_curve(ind_labels, ind_preds)
    
    axes[0].plot(fpr_test, tpr_test, linewidth=2, label=f'Original Test (AUC={test_auc:.4f})', color='#2E86AB')
    axes[0].plot(fpr_ind, tpr_ind, linewidth=2, label=f'Independent (AUC={ind_auc:.4f})', color='#A23B72')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curves - Generalization Test', fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=11)
    axes[0].grid(alpha=0.3)
    
    # Metrics comparison
    metrics = ['ROC-AUC', 'Sensitivity', 'Specificity']
    test_vals = [test_auc, test_sensitivity, test_specificity]
    ind_vals = [ind_auc, ind_sensitivity, ind_specificity]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1].bar(x - width/2, test_vals, width, label='Original Test', color='#2E86AB', alpha=0.8)
    axes[1].bar(x + width/2, ind_vals, width, label='Independent', color='#A23B72', alpha=0.8)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Performance Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].legend(fontsize=11)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim([0.7, 1.0])
    
    # Add value labels on bars
    for i, (tv, iv) in enumerate(zip(test_vals, ind_vals)):
        axes[1].text(i - width/2, tv + 0.01, f'{tv:.3f}', ha='center', fontsize=9)
        axes[1].text(i + width/2, iv + 0.01, f'{iv:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/generalization_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plots saved")
    
    # ============================================================
    # FINAL ASSESSMENT
    # ============================================================
    print("\n" + "="*60)
    print("✅ GENERALIZATION ASSESSMENT")
    print("="*60)
    
    auc_drop = test_auc - ind_auc
    
    if abs(auc_drop) < 0.02:
        assessment = "🌟 EXCELLENT - Model generalizes perfectly to independent data!"
    elif abs(auc_drop) < 0.05:
        assessment = "✅ GOOD - Minor performance variation, acceptable generalization."
    elif abs(auc_drop) < 0.10:
        assessment = "⚠️  MODERATE - Some generalization gap, consider domain adaptation."
    else:
        assessment = "❌ POOR - Significant generalization gap, model may be overfit."
    
    print(f"\n{assessment}")
    print(f"\nAUC Change: {test_auc:.4f} → {ind_auc:.4f} (Δ = {auc_drop:+.4f})")
    print(f"Sensitivity: {test_sensitivity:.4f} → {ind_sensitivity:.4f}")
    print(f"Specificity: {test_specificity:.4f} → {ind_specificity:.4f}")
    
    print("\n" + "="*60)
    
    return {
        'independent': {
            'auc': ind_auc,
            'sensitivity': ind_sensitivity,
            'specificity': ind_specificity,
            'f1': ind_f1
        },
        'test': {
            'auc': test_auc,
            'sensitivity': test_sensitivity,
            'specificity': test_specificity
        }
    }

if __name__ == "__main__":
    # First prepare the independent dataset
    print("Step 1: Preparing independent dataset...")
    from prepare_independent_dataset import prepare_da_db_dataset
    
    df = prepare_da_db_dataset()
    
    if df is not None and len(df) > 0:
        print("\nStep 2: Running evaluation...")
        results = evaluate_independent_dataset()
    else:
        print("\n❌ Could not prepare independent dataset.")
        print("Please check the DA+DB dataset location.")
