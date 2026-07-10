"""
model_htnet.py -- Lightweight, HTNet-inspired model. Each of the 4 facial ROI
patches (28x28x5: RGB+Flow) goes through a SMALL CNN (not ResNet18 -- far
fewer parameters, better matched to ~680 training samples). The 4 resulting
region embeddings are combined via a small Transformer encoder layer to model
interactions between regions (e.g. eyes + mouth moving together), then pooled
and classified.
"""
import torch
import torch.nn as nn

class SmallPatchCNN(nn.Module):
    """Tiny CNN for a single 28x28x5 patch -- deliberately small to avoid
    overfitting on limited data, unlike the ResNet18 backbones used before."""
    def __init__(self, in_channels=5, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7

            nn.Conv2d(32, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # -> (embed_dim, 1, 1), works regardless of input H/W
        )

    def forward(self, x):
        feat = self.net(x)
        return feat.flatten(1)

class HTNetLite(nn.Module):
    def __init__(self, num_classes, embed_dim=64, num_regions=3, dropout=0.3):
        super().__init__()
        self.patch_cnn = SmallPatchCNN(in_channels=5, embed_dim=embed_dim)

        self.region_pos_embed = nn.Parameter(torch.zeros(1, num_regions, embed_dim))
        nn.init.trunc_normal_(self.region_pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=embed_dim * 2,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, patches):
        # patches: (B, 4, 5, 28, 28)
        B, R, C, H, W = patches.shape
        patches_flat = patches.view(B * R, C, H, W)
        embeds = self.patch_cnn(patches_flat)
        embeds = embeds.view(B, R, -1)

        embeds = embeds + self.region_pos_embed

        fused = self.transformer(embeds)
        pooled = fused.mean(dim=1)

        return self.classifier(pooled)