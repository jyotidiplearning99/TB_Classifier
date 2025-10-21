# src/eval_no_clahe.py

import os
PROJECT_DIR = "/scratch/project_2010751/TB_Classifier"
os.environ['HF_HOME'] = f"{PROJECT_DIR}/.cache/huggingface"
os.environ['TORCH_HOME'] = f"{PROJECT_DIR}/.cache/torch"

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix

from model import MedicalTBClassifier
from dataset_no_clahe import TBDatasetNoCLAHE, get_valid_transforms

def evaluate_no_clahe():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*60)
    print("🧪 TESTING WITHOUT CLAHE")
    print("="*60)
    
    # Load model
    checkpoint = torch.load('outputs/tb_sota_final/best_model.pth', 
                           map_location=device, weights_only=False)
    
    model = MedicalTBClassifier(
        model_name='convnextv2_base.fcmae_ft_in22k_in1k',
        pretrained=False
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    
    # Apply EMA
    ema_shadow = checkpoint.get('ema_shadow', {})
    if ema_shadow:
        state_dict = model.state_dict()
        for name, shadow_param in ema_shadow.items():
            if name in state_dict:
                state_dict[name] = shadow_param
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    
    val_threshold = checkpoint['metrics']['threshold']
    
    # Test WITH CLAHE (current)
    print("\n1️⃣ Testing WITH CLAHE (current pipeline):")
    
    dataset_with = TBDatasetNoCLAHE(
        csv_file='metadata/independent_test.csv',
        transform=get_valid_transforms(512),
        use_clahe=True  # Current behavior
    )
    
    loader_with = torch.utils.data.DataLoader(
        dataset_with, batch_size=16, shuffle=False, num_workers=2
    )
    
    preds_with, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader_with, desc="With CLAHE"):
            images = batch['image'].to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            preds_with.extend(probs)
            labels.extend(batch['label'].numpy())
    
    preds_with = np.array(preds_with)
    labels = np.array(labels)
    
    auc_with = roc_auc_score(labels, preds_with)
    binary_with = (preds_with >= val_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, binary_with, labels=[0,1]).ravel()
    
    print(f"  AUC:         {auc_with:.4f}")
    print(f"  Sensitivity: {tp/(tp+fn)*100:.1f}%")
    print(f"  Mean prob:   {preds_with.mean():.4f}")
    
    # Test WITHOUT CLAHE
    print("\n2️⃣ Testing WITHOUT CLAHE (test hypothesis):")
    
    dataset_without = TBDatasetNoCLAHE(
        csv_file='metadata/independent_test.csv',
        transform=get_valid_transforms(512),
        use_clahe=False  # NEW: No CLAHE
    )
    
    loader_without = torch.utils.data.DataLoader(
        dataset_without, batch_size=16, shuffle=False, num_workers=2
    )
    
    preds_without = []
    with torch.no_grad():
        for batch in tqdm(loader_without, desc="Without CLAHE"):
            images = batch['image'].to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            preds_without.extend(probs)
    
    preds_without = np.array(preds_without)
    
    auc_without = roc_auc_score(labels, preds_without)
    binary_without = (preds_without >= val_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, binary_without, labels=[0,1]).ravel()
    
    print(f"  AUC:         {auc_without:.4f}")
    print(f"  Sensitivity: {tp/(tp+fn)*100:.1f}%")
    print(f"  Mean prob:   {preds_without.mean():.4f}")
    
    # Comparison
    print("\n" + "="*60)
    print("📊 COMPARISON")
    print("="*60)
    print(f"AUC difference:  {auc_without - auc_with:+.4f}")
    print(f"Mean prob diff:  {preds_without.mean() - preds_with.mean():+.4f}")
    
    if auc_without > auc_with + 0.02:
        print("\n✅ WITHOUT CLAHE IS BETTER!")
        print("   → DA+DB images are already contrast-enhanced")
    elif auc_with > auc_without + 0.02:
        print("\n✅ WITH CLAHE IS BETTER")
        print("   → Keep CLAHE in pipeline")
    else:
        print("\n⚠️  NO SIGNIFICANT DIFFERENCE")
        print("   → CLAHE not the issue")
    
    print("="*60 + "\n")
    
    return auc_with, auc_without

if __name__ == "__main__":
    auc_with, auc_without = evaluate_no_clahe()
