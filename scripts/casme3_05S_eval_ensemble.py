import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, accuracy_score

from casme3_dataset import fit_label_map, build_dataloader
import torchvision.models as models

# --- SHARED BACKBONE UTILS ---

def make_backbone(in_channels):
    backbone = models.resnet18(weights=None)
    if in_channels != 3:
        old_conv1 = backbone.conv1
        new_conv1 = nn.Conv2d(in_channels, old_conv1.out_channels, kernel_size=old_conv1.kernel_size,
                               stride=old_conv1.stride, padding=old_conv1.padding, bias=False)
        backbone.conv1 = new_conv1
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feat_dim

# --- MODEL 1: ThreeStreamModelV2 (Champion 0.3105) ---

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

# --- MODEL 2: TwoStreamModel (Exp C 0.2692) ---

class TwoStreamModel(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        self.rgb_backbone, rgb_dim = make_backbone(3)
        self.flow_backbone, flow_dim = make_backbone(2)

        fused_dim = rgb_dim + flow_dim
        
        # NOTE: TwoStreamModel used 256 for hidden layer, matching casme3_model.py
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, rgb, flow, depth_ignored=None):
        rgb_feat = self.rgb_backbone(rgb)
        flow_feat = self.flow_backbone(flow)
        fused = torch.cat([rgb_feat, flow_feat], dim=1)
        return self.classifier(fused)

# --- EVALUATION ---

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MANIFEST_PATH = Path("manifest_casme3_767.csv")
PROCESSED_DIR = Path("processed")

@torch.no_grad()
def evaluate_ensemble(fold, val_loader, model_champ, model_expc):
    model_champ.eval()
    model_expc.eval()
    
    all_preds, all_labels = [], []
    
    for rgb, flow, depth, y in tqdm(val_loader, desc=f"  Val Fold {fold}", leave=False):
        rgb, flow, depth = rgb.to(DEVICE), flow.to(DEVICE), depth.to(DEVICE)
        
        logits_champ = model_champ(rgb, flow, depth)
        logits_expc = model_expc(rgb, flow) # depth ignored
        
        # Softmax Averaging
        probs_champ = torch.softmax(logits_champ, dim=1)
        probs_expc = torch.softmax(logits_expc, dim=1)
        
        avg_probs = (probs_champ + probs_expc) / 2.0
        preds = avg_probs.argmax(1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.numpy())
        
    uf1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    uar = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return uf1, uar

def main():
    print("\n=== Softmax Ensemble Evaluation: Champion + Exp C ===\n")
    
    df = pd.read_csv(MANIFEST_PATH)
    if 'clip_name' not in df.columns and 'clip_id' in df.columns:
        df['clip_name'] = df['clip_id']
    if 'fold' not in df.columns and 'fold_number' in df.columns:
        df['fold'] = df['fold_number']
        
    df = df[df['clip_name'].apply(lambda x: (PROCESSED_DIR / f"{x}.npz").exists())].copy()
    df["emotion"] = df["emotion"].str.strip().str.lower()
    label_map = fit_label_map(df["emotion"])
    num_classes = len(label_map)
    
    results = {}
    
    for fold in sorted(df["fold"].unique()):
        val_df = df[df["fold"] == fold].reset_index(drop=True)
        val_loader = build_dataloader(val_df, label_map, batch_size=32, shuffle=False, is_train=False)
        
        champ_path = Path(f"casme3_best_threestream_v2_fold{fold}.pt")
        expc_path = Path(f"casme3_best_twostream_fold{fold}.pt")
        
        if not champ_path.exists() or not expc_path.exists():
            print(f"Skipping fold {fold} - missing checkpoints.")
            continue
            
        model_champ = ThreeStreamModelV2(num_classes=num_classes).to(DEVICE)
        model_champ.load_state_dict(torch.load(champ_path, map_location=DEVICE, weights_only=True))
        
        model_expc = TwoStreamModel(num_classes=num_classes).to(DEVICE)
        model_expc.load_state_dict(torch.load(expc_path, map_location=DEVICE, weights_only=True))
        
        uf1, uar = evaluate_ensemble(fold, val_loader, model_champ, model_expc)
        results[fold] = {'uf1': uf1, 'uar': uar}
        print(f"Fold {fold}: UF1={uf1:.4f}, UAR={uar:.4f}")
        
    if results:
        uf1s = [m['uf1'] for m in results.values()]
        uars = [m['uar'] for m in results.values()]
        
        print(f"\n{'='*60}\nFINAL ENSEMBLE SUMMARY\n{'='*60}")
        print(f"  Mean UF1: {np.mean(uf1s):.4f} ± {np.std(uf1s):.4f}")
        print(f"  Mean UAR: {np.mean(uars):.4f} ± {np.std(uars):.4f}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
