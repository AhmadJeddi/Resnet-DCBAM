"""Small runnable example that imports the backbone and runs a forward pass.

Usage:
    python examples/test_backbone.py
"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from models.backbone.resnet_dcbam import BackboneWithDCBAM

device = "cuda" if torch.cuda.is_available() else "cpu"
freeze_layers = {'conv1': True, 'bn1': True,
                'layer1': True, 'layer2': True,
                'layer3': True, 'layer4': False}
model = BackboneWithDCBAM(freeze_layers=freeze_layers).to(device)
model.eval()

# Forward pass
B, C, H, W = 2, 3, 448, 448
x = torch.randn(B, C, H, W, device=device)
with torch.no_grad():
    feats = model(x)

print("Feature shapes:")
for k, f in feats.items():
    print(f"  {k}: {tuple(f.shape)}")

# Count frozen and trainable params
total_params = sum(p.numel() for p in model.parameters())
frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"Frozen {frozen_params}/{total_params} ({frozen_params/total_params*100:.2f}%)")

# DCBAM params
dcbam_params = [p for n, p in model.named_parameters() if "dcbam" in n]
dcbam_total = sum(p.numel() for p in dcbam_params)
dcbam_frozen = sum(p.numel() for p in dcbam_params if not p.requires_grad)
dcbam_trainable = dcbam_total - dcbam_frozen
print(f"DCBAM total params: {dcbam_total}, frozen: {dcbam_frozen}, trainable: {dcbam_trainable}")
