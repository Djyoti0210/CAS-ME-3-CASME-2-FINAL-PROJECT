import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from casme3_dataset import fit_label_map, build_dataloader
import torchvision.models as models

# --- PURE FEATURE EXTRACTOR ---

def make_resnet_backbone(in_channels):
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
    
    # We only care about extracting the 512-D features, so we replace the fc layer.
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    
    # Strictly lock the backbone in evaluation mode
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
        
    return backbone, feat_dim

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_backbone, rgb_dim = make_resnet_backbone(3)
        self.flow_backbone, flow_dim = make_resnet_backbone(2)
        self.depth_backbone, depth_dim = make_resnet_backbone(1)
        
        self.fused_dim = rgb_dim + flow_dim + depth_dim

    def forward(self, rgb, flow, depth):
        r = self.rgb_backbone(rgb)
        f = self.flow_backbone(flow)
        d = self.depth_backbone(depth)
        return torch.cat([r, f, d], dim=1)

# --- UTILITIES ---

MANIFEST_PATH = Path("manifest_casme3_767.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METRICS_DIR = Path("metrics")

def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def plot_confusion_matrix(cm, class_names, fold):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'SVM Confusion Matrix - Fold {fold}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(METRICS_DIR / f"fold_{fold}_svm_cm.png", dpi=150)
    plt.close()

def extract_all_features(df, label_map):
    print("\n--- PHASE 1: DEEP LEARNING FEATURE EXTRACTION ---")
    model = FeatureExtractor().to(DEVICE)
    model.eval()
    
    # Dataloader without shuffling to keep order identical to df
    loader = build_dataloader(df, label_map, batch_size=32, shuffle=False, is_train=False, use_weighted_sampler=False)
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for rgb, flow, depth, y in tqdm(loader, desc="Extracting 1536-D Features", unit="batch"):
            rgb, flow, depth = rgb.to(DEVICE), flow.to(DEVICE), depth.to(DEVICE)
            feat = model(rgb, flow, depth)
            all_features.append(feat.cpu().numpy())
            all_labels.append(y.numpy())
            
    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    return X, y

def train_svm_kfold(df, X, y, label_map):
    print("\n--- PHASE 2: SVM MACHINE LEARNING FUSION ---")
    
    results = {}
    class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    
    for fold in sorted(df["fold"].unique()):
        print(f"\n{'='*40}\nFOLD {fold}\n{'='*40}")
        
        train_mask = df["fold"] != fold
        val_mask = df["fold"] == fold
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        
        # 1. Feature Standardization (Critical for SVMs)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # 2. Support Vector Machine with Mathematical Class Balancing
        # RBF Kernel acts as an infinite-dimensional feature space map
        # C=1.0 is standard regularization
        # class_weight='balanced' perfectly handles the massive 'others' class imbalance
        clf = SVC(kernel='rbf', C=1.0, class_weight='balanced', random_state=42)
        
        print("Training SVM... ", end="")
        clf.fit(X_train, y_train)
        print("Done!")
        
        # 3. Predict and Evaluate
        preds = clf.predict(X_val)
        
        uf1 = f1_score(y_val, preds, average='macro', zero_division=0)
        uar = recall_score(y_val, preds, average='macro', zero_division=0)
        acc = accuracy_score(y_val, preds)
        
        results[fold] = {'uf1': uf1, 'uar': uar, 'acc': acc}
        print(f"Fold {fold} Results -> ACC: {acc:.4f} | UF1: {uf1:.4f} | UAR: {uar:.4f}")
        
        # Save Confusion Matrix
        cm = confusion_matrix(y_val, preds, labels=range(len(label_map)))
        plot_confusion_matrix(cm, class_names, fold)

    # Calculate Summaries
    uf1s = [m['uf1'] for m in results.values()]
    uars = [m['uar'] for m in results.values()]
    
    print(f"\n{'='*60}\nFINAL SVM BYPASS SUMMARY\n{'='*60}")
    for fold, m in results.items():
        print(f"  Fold {fold}: UF1={m['uf1']:.4f}, UAR={m['uar']:.4f}")
    print("-" * 40)
    print(f"  Mean UF1: {np.mean(uf1s):.4f} ± {np.std(uf1s):.4f}")
    print(f"  Mean UAR: {np.mean(uars):.4f} ± {np.std(uars):.4f}")
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    seed_everything(42)
    METRICS_DIR.mkdir(exist_ok=True)
    
    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        return

    df = pd.read_csv(MANIFEST_PATH)
    if 'clip_name' not in df.columns and 'clip_id' in df.columns:
        df['clip_name'] = df['clip_id']
    if 'fold' not in df.columns and 'fold_number' in df.columns:
        df['fold'] = df['fold_number']
        
    PROCESSED_DIR = Path("processed")
    # Ensure dataset matches the processed files
    df = df[df['clip_name'].apply(lambda x: (PROCESSED_DIR / f"{x}.npz").exists())].copy()
    
    df["emotion"] = df["emotion"].str.strip().str.lower()
    label_map = fit_label_map(df["emotion"])
    print("Label map:", label_map)
    print("Device:", DEVICE)
    print(f"Total valid samples: {len(df)}")

    # Execute the SVM Bypass strategy
    X, y = extract_all_features(df, label_map)
    train_svm_kfold(df, X, y, label_map)

if __name__ == "__main__":
    main()
