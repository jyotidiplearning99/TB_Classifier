# src/eval_alt_normalization.py

import os
PROJECT_DIR = "/scratch/project_2010751/TB_Classifier"
os.environ['HF_HOME'] = f"{PROJECT_DIR}/.cache/huggingface"
os.environ['TORCH_HOME'] = f"{PROJECT_DIR}/.cache/torch"

import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score

from model import MedicalTBClassifier
from dataset_no_clahe import TBDatasetNoCLAHE

def get_alt_normalization():
    """Alternative normalization for DA+DB"""
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(
            mean=[0.5, 0.5, 0.5],  # Changed from [0.485, 0.456, 0.406]
            std=[0.25, 0.25, 0.25]  # Changed from [0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def test_normalization():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*60)
    print("🔬 TESTING ALTERNATIVE NORMALIZATION")
    print("="*60)
    
    # Load model
    checkpoint = torch.load('outputs/tb_sota_final/best_model.pth', 
                           map_location=device, weights_only=False)
    
    model = MedicalTBClassifier(
        model_name='convnextv2_base.fcmae_ft_in22k_in1k',
        pretrained=False
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # EMA
    if 'ema_shadow' in checkpoint:
        sd = model.state_dict()
        for k, v in checkpoint['ema_shadow'].items():
            if k in sd:
                sd[k] = v
        model.load_state_dict(sd)
    
    model.eval()
    
    # Test with alternative normalization
    dataset = TBDatasetNoCLAHE(
        csv_file='metadata/independent_test.csv',
        transform=get_alt_normalization(),
        use_clahe=False
    )
    
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=2
    )
    
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            preds.extend(probs)
            labels.extend(batch['label'].numpy())
    
    preds = np.array(preds)
    labels = np.array(labels)
    
    auc = roc_auc_score(labels, preds)
    
    print(f"\nWith Alternative Normalization:")
    print(f"  mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]")
    print(f"  AUC: {auc:.4f}")
    print(f"  Mean prob: {preds.mean():.4f}")
    print(f"  Median prob: {np.median(preds):.4f}")
    
    print("\n" + "="*60)
    
    return auc

if __name__ == "__main__":
    test_normalization()
