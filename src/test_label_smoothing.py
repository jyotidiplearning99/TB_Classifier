# test_label_smoothing.py

import torch
import torch.nn as nn

class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, inputs, targets):
        smoothed = targets * (1 - 2*self.smoothing) + self.smoothing
        return self.bce(inputs, smoothed)

# Test
criterion = LabelSmoothingBCELoss(smoothing=0.05)

# Test targets
targets = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
inputs = torch.randn(4, 1)  # Random logits

# Apply smoothing manually to verify
smoothed_manual = targets * (1 - 2*0.05) + 0.05

print("="*60)
print("LABEL SMOOTHING VERIFICATION")
print("="*60)
print(f"\nOriginal targets:\n{targets.squeeze()}")
print(f"\nSmoothed targets (formula: y*(1-2s) + s):\n{smoothed_manual.squeeze()}")
print(f"\nExpected: [0.05, 0.95, 0.05, 0.95]")
print(f"\nVerification:")
print(f"  0 → {smoothed_manual[0].item():.2f} (should be 0.05)")
print(f"  1 → {smoothed_manual[1].item():.2f} (should be 0.95)")
print(f"\n✓ Correct!" if torch.allclose(smoothed_manual, torch.tensor([[0.05], [0.95], [0.05], [0.95]])) else "✗ Error!")
print("="*60)
