"""
model_temporal.py -- CNN (per-frame feature extractor, ResNet18) + GRU
(temporal aggregation) model. Simpler and faster to train than a full 3D-CNN,
while still capturing motion dynamics across the whole clip.
"""
import torch
import torch.nn as nn
import torchvision.models as models

class CNNGRUModel(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, dropout=0.5):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.cnn = backbone  # shared across all frames (applied per-frame)

        self.gru = nn.GRU(input_size=feat_dim, hidden_size=hidden_dim,
                           num_layers=1, batch_first=True, bidirectional=True)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim * 2, 128),  # *2 for bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, frames):
        # frames: (B, T, 3, H, W)
        B, T, C, H, W = frames.shape
        frames_flat = frames.view(B * T, C, H, W)
        feats = self.cnn(frames_flat)               # (B*T, feat_dim)
        feats = feats.view(B, T, -1)                 # (B, T, feat_dim)

        gru_out, _ = self.gru(feats)                 # (B, T, hidden_dim*2)
        pooled = gru_out.mean(dim=1)                 # mean pool over time -- (B, hidden_dim*2)

        return self.classifier(pooled)
