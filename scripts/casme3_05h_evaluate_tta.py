import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, accuracy_score
import torchvision.models as models
from dataset import fit_label_map, build_dataloader

def make_backbone(in_channels):
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if in_channels != 3:
        old_conv1 = backbone.conv1
        new_conv1 = nn.Conv2d(in_channels, old_conv1.out_channels, kernel_size=old_conv1.kernel_size,
                               stride=old_conv1.stride, padding=old_conv1.padding, bias=False)
        with torch.no_grad():
            if in_channels == 1:
                new_conv1.weight[:, 0] = old_conv1.weight.mean(dim=1)
            elif in_channels == 2:
                new_conv1.weight[:, 0] = old_conv1.weight[:, 0]
                new_conv1.weight[:, 1] = old_conv1.weight[:, 1]
        backbone.conv1 = new_conv1
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feat_dim

class ThreeStreamModelV2(nn.Module):
    def __init__(self, num_classes, dropout=0.4):
        super().__init__()
        self.rgb_backbone, rgb_dim = make_backbone(3)
        self.flow_backbone, flow_dim = make_backbone(2)
        self.depth_backbone, depth_dim = make_backbone(1)

        fused_dim = rgb_dim + flow_dim + depth_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, rgb, flow, depth):
        rgb_feat = self.rgb_backbone(rgb)
        flow_feat = self.flow_backbone(flow)
        depth_feat = self.depth_backbone(depth)
        fused = torch.cat([rgb_feat, flow_feat, depth_feat], dim=1)
        return self.classifier(fused)

MANIFEST_PATH = Path("CASME3_Dataset/fold_manifest.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def evaluate_tta(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for rgb, flow, depth, y in tqdm(loader, desc="  val tta  ", leave=False):
        rgb, flow, depth, y = rgb.to(DEVICE), flow.to(DEVICE), depth.to(DEVICE), y.to(DEVICE)
        
        # Original Forward
        logits_orig = model(rgb, flow, depth)
        
        # Flipped Forward
        rgb_flip = torch.flip(rgb, dims=[3])
        flow_flip = torch.flip(flow, dims=[3])
        flow_flip[:, 0, :, :] = -flow_flip[:, 0, :, :] # Invert X-axis flow
        depth_flip = torch.flip(depth, dims=[3])
        
        logits_flip = model(rgb_flip, flow_flip, depth_flip)
        
        # Ensemble Average
        logits = (logits_orig + logits_flip) / 2.0
        
        preds = logits.argmax(1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        
    uf1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    uar = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    
    return acc, uf1, uar

def main():
    df = pd.read_csv(MANIFEST_PATH)
    if 'clip_name' not in df.columns and 'clip_id' in df.columns: df['clip_name'] = df['clip_id']
    if 'fold' not in df.columns and 'fold_number' in df.columns: df['fold'] = df['fold_number']
        
    PROCESSED_DIR = Path("processed")
    df = df[df['clip_name'].apply(lambda x: (PROCESSED_DIR / f"{x}.npz").exists())].copy()
    
    df["emotion"] = df["emotion"].str.strip().str.lower()
    label_map = fit_label_map(df["emotion"])
    
    print("\n=== EVALUATING EXP L: HORIZONTAL FLIP TTA ON CHAMPION ===\n")
    results = {}
    
    for fold in sorted(df["fold"].unique()):
        print(f"FOLD {fold}")
        val_df = df[df["fold"] == fold].reset_index(drop=True)
        val_loader = build_dataloader(val_df, label_map, batch_size=16, shuffle=False, is_train=False)
        
        model = ThreeStreamModelV2(num_classes=len(label_map)).to(DEVICE)
        ckpt_path = Path(f"best_threestream_v2_fold{fold}.pt")
        
        if not ckpt_path.exists():
            print(f"  [ERROR] {ckpt_path} not found. Skipping fold.")
            continue
            
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        
        acc, uf1, uar = evaluate_tta(model, val_loader)
        results[fold] = {'uf1': uf1, 'uar': uar}
        print(f"  UF1={uf1:.4f}, UAR={uar:.4f}")
        
    uf1s = [m['uf1'] for m in results.values()]
    uars = [m['uar'] for m in results.values()]
    
    print(f"\n{'='*60}\nFINAL SUMMARY (Experiment L: TTA)\n{'='*60}")
    for fold, m in results.items():
        print(f"  Fold {fold}: UF1={m['uf1']:.4f}, UAR={m['uar']:.4f}")
    print("-" * 40)
    print(f"  Mean UF1: {np.mean(uf1s):.4f} ± {np.std(uf1s):.4f}")
    print(f"  Mean UAR: {np.mean(uars):.4f} ± {np.std(uars):.4f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
