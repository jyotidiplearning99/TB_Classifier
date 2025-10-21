# src/model_medical.py

import os
import sys

# ⚠️ SET CACHE FIRST - BEFORE ANY IMPORTS
PROJECT_DIR = "/scratch/project_2010751/TB_Classifier"
os.environ['HF_HOME'] = f"{PROJECT_DIR}/.cache/huggingface"
os.environ['TORCH_HOME'] = f"{PROJECT_DIR}/.cache/torch"
os.environ['TRANSFORMERS_CACHE'] = f"{PROJECT_DIR}/.cache/huggingface"
os.environ['HUGGINGFACE_HUB_CACHE'] = f"{PROJECT_DIR}/.cache/huggingface/hub"

# Create cache directories immediately
for cache_dir in [os.environ['HF_HOME'], os.environ['TORCH_HOME'], 
                  os.environ['TRANSFORMERS_CACHE'], os.environ['HUGGINGFACE_HUB_CACHE']]:
    os.makedirs(cache_dir, exist_ok=True)

print(f"✓ Cache set to: {os.environ['HF_HOME']}")

# NOW import torch and timm
import torch
import torch.nn as nn
import timm

class MedicalTBClassifier(nn.Module):
    """
    TB Classifier with consistent TIMM backbone interface
    Uses features_only=True for compatibility across all architectures
    """
    
    def __init__(self, 
                 model_name='convnextv2_base.fcmae_ft_in22k_in1k',
                 pretrained=True,
                 num_classes=1,
                 dropout=0.3):
        super().__init__()
        
        print(f"✓ Loading backbone: {model_name}")
        
        # Create backbone with features_only for consistent output
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,    # Return feature maps
            out_indices=[-1]       # Last stage only
        )
        
        # Get feature dimension from feature_info
        self.feature_dim = self.backbone.feature_info[-1]['num_chs']
        
        print(f"✓ Feature dimension: {self.feature_dim}")
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Extract features - returns list, take first (and only) element
        features = self.backbone(x)[0]  # (B, C, H, W)
        
        # Global average pooling
        pooled = self.global_pool(features).flatten(1)  # (B, C)
        
        # Classify
        output = self.classifier(pooled)
        
        return output

# Test model - FIXED
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 TESTING MODEL")
    print("="*60 + "\n")
    
    print("⚠️  Running lightweight test (no weight loading)")
    print("    For full test, run on compute node with srun/sbatch\n")
    
    # Corrected model names
    backbones = [
        'convnextv2_base.fcmae_ft_in22k_in1k',
        'efficientnetv2_rw_m.agc_in1k',
        # 'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k',
    ]
    
    for backbone_name in backbones:
        print(f"\nTesting: {backbone_name}")
        try:
            # Test WITHOUT loading pretrained weights
            model = MedicalTBClassifier(
                model_name=backbone_name,
                pretrained=False  # Don't download weights for test
            )
            
            # ✅ FIX: Put model in eval mode OR use batch_size > 1
            model.eval()  # This disables BatchNorm training behavior
            
            # Test with batch_size=2 (also works with batch_size=1 in eval mode)
            x = torch.randn(2, 3, 224, 224)  # Batch size 2
            
            with torch.no_grad():  # No gradients needed for test
                out = model(x)
            
            print(f"✓ Input shape: {x.shape}")
            print(f"✓ Output shape: {out.shape}")
            print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("✓ Lightweight test complete")
    print("  To test with pretrained weights, run:")
    print("="*60 + "\n")
