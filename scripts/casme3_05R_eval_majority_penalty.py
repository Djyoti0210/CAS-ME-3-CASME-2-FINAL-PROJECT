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

from casme3_dataset import fit_label_map, build_dataloader
import torchvision.models as models

# --- REBUILD ARCHITECTURE TO LOAD 0.3105 CHECKPOINTS ---

def make_backbone(in_channels):
    backbone = models.resnet18(weights=None) # We will load weights, so no need for ImageNet download
    if in_channels != 3:
        old_conv1 = backbone.conv1
        new_conv1 = nn.Conv2d(in_channels, old_conv1.out_channels, kernel_size=old_conv1.kernel_size,
                               stride=old_conv1.stride, padding=old_conv1.padding, bias=False)
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

# --- UTILS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MANIFEST_PATH = Path("manifest_casme3_767.csv")
PROCESSED_DIR = Path("processed")

@torch.no_grad()
def extract_logits(model, loader):
    model.eval()
    all_logits = []
    all_labels = []
    for rgb, flow, depth, y in tqdm(loader, desc="  Extracting Logits", leave=False):
        rgb, flow, depth = rgb.to(DEVICE), flow.to(DEVICE), depth.to(DEVICE)
        logits = model(rgb, flow, depth)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)

def main():
    print("=== Inference-Time Majority-Class Penalization (Logit Correction) ===")
    
    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        return

    df = pd.read_csv(MANIFEST_PATH)
    if 'clip_name' not in df.columns and 'clip_id' in df.columns:
        df['clip_name'] = df['clip_id']
    if 'fold' not in df.columns and 'fold_number' in df.columns:
        df['fold'] = df['fold_number']
        
    df = df[df['clip_name'].apply(lambda x: (PROCESSED_DIR / f"{x}.npz").exists())].copy()
    df["emotion"] = df["emotion"].str.strip().str.lower()
    label_map = fit_label_map(df["emotion"])
    num_classes = len(label_map)
    print("Label map:", label_map)
    
    penalties_to_test = [0.0, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0, 2.0]
    
    # Store results for each penalty
    results_by_penalty = {p: {'uf1': [], 'uar': []} for p in penalties_to_test}

    for fold in sorted(df["fold"].unique()):
        print(f"\n{'='*40}\nFOLD {fold}\n{'='*40}")
        
        ckpt_path = Path(f"casme3_best_threestream_v2_fold{fold}.pt")
        if not ckpt_path.exists():
            print(f"Skipping fold {fold}: checkpoint {ckpt_path} not found.")
            continue
            
        train_df = df[df["fold"] != fold].reset_index(drop=True)
        val_df = df[df["fold"] == fold].reset_index(drop=True)
        
        # Calculate training class frequencies to determine the penalty vector
        counts = train_df["emotion"].value_counts()
        freqs = np.zeros(num_classes)
        for cls, idx in label_map.items():
            freqs[idx] = counts.get(cls, 0) / len(train_df)
            
        print(f"Training Class Frequencies: {np.round(freqs, 3)}")
        
        # Build dataloader for validation
        val_loader = build_dataloader(val_df, label_map, batch_size=32, shuffle=False, is_train=False)
        
        # Load model and weights
        model = ThreeStreamModelV2(num_classes=num_classes).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        
        # Extract raw logits for this fold's validation set
        logits, labels = extract_logits(model, val_loader)
        
        # Grid Search Inference-Time Penalties
        for penalty_strength in penalties_to_test:
            # The penalty is proportional to the training frequency of the class
            # e.g., 'others' might have freq 0.60, 'anger' freq 0.05
            # We subtract (penalty_strength * freq) from the logit before argmax
            penalty_vector = penalty_strength * freqs
            corrected_logits = logits - penalty_vector
            
            preds = np.argmax(corrected_logits, axis=1)
            
            uf1 = f1_score(labels, preds, average='macro', zero_division=0)
            uar = recall_score(labels, preds, average='macro', zero_division=0)
            
            results_by_penalty[penalty_strength]['uf1'].append(uf1)
            results_by_penalty[penalty_strength]['uar'].append(uar)
            
            if penalty_strength == 0.0:
                print(f"Baseline (Penalty 0.0) -> UF1: {uf1:.4f}, UAR: {uar:.4f}")
                
    print(f"\n{'='*60}\nFINAL GRID SEARCH RESULTS (5-Fold Mean)\n{'='*60}")
    for p in penalties_to_test:
        mean_uf1 = np.mean(results_by_penalty[p]['uf1'])
        mean_uar = np.mean(results_by_penalty[p]['uar'])
        
        marker = " (Baseline)" if p == 0.0 else ""
        print(f"Penalty {p:<4} -> Mean UF1: {mean_uf1:.4f}, Mean UAR: {mean_uar:.4f}{marker}")

if __name__ == "__main__":
    main()
