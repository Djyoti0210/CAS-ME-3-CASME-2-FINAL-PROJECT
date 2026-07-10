import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, recall_score, confusion_matrix, accuracy_score
import torchvision.models as models

# --- DATASET ---

PROCESSED_DIR = Path("processed")
OFFSET_DIR = Path("CASME3_Dataset/processed_offset")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def fit_label_map(emotions):
    unique = sorted(emotions.unique())
    return {e: i for i, e in enumerate(unique)}

class DualFlowDataset(Dataset):
    def __init__(self, df, label_map, is_train=False):
        self.df = df.reset_index(drop=True)
        self.label_map = label_map
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clip_name = row['clip_name']
        
        # 1. RGB and Depth
        npz_path = PROCESSED_DIR / f"{clip_name}.npz"
        data = np.load(npz_path)
        rgb = data["rgb"].copy().astype(np.float32) / 255.0
        depth = data["depth"].copy().astype(np.float32) / 255.0
        depth = depth[..., None]
        
        # 2. Flow Onset->Apex (2 channels)
        flow_apex = data["flow"].copy()
        flow_apex = np.clip(flow_apex / 15.0, -1.0, 1.0)
        
        # 3. Flow Onset->Offset (2 channels)
        try:
            off_data = np.load(OFFSET_DIR / f"{clip_name}.npz")
            flow_offset = off_data["flow_offset"].copy()
            flow_offset = np.clip(flow_offset / 15.0, -1.0, 1.0)
        except Exception:
            flow_offset = np.zeros_like(flow_apex)
            
        # Stack to 4 channels: (H, W, 4)
        flow_dual = np.concatenate([flow_apex, flow_offset], axis=-1)

        if self.is_train:
            import random
            if random.random() > 0.5:
                rgb = np.flip(rgb, axis=1).copy()
                depth = np.flip(depth, axis=1).copy()
                
                flow_dual = np.flip(flow_dual, axis=1).copy()
                flow_dual[..., 0] = -flow_dual[..., 0] # Apex X
                flow_dual[..., 2] = -flow_dual[..., 2] # Offset X

        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD

        rgb_t = torch.from_numpy(rgb).float().permute(2, 0, 1)
        flow_t = torch.from_numpy(flow_dual).float().permute(2, 0, 1)
        depth_t = torch.from_numpy(depth).float().permute(2, 0, 1)

        label = self.label_map[str(row["emotion"]).strip().lower()]
        return rgb_t, flow_t, depth_t, torch.tensor(label, dtype=torch.long)

def build_dataloader(df, label_map, batch_size=16, shuffle=True, is_train=False, use_weighted_sampler=True):
    dataset = DualFlowDataset(df, label_map, is_train=is_train)
    sampler = None
    if shuffle and is_train and use_weighted_sampler:
        class_counts = df["emotion"].value_counts()
        total = len(df)
        weights_dict = {cls: total / count for cls, count in class_counts.items()}
        sample_weights = [weights_dict[str(e).strip().lower()] for e in df["emotion"]]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=0, pin_memory=True)

# --- MODEL DEFINITION ---

def make_backbone(in_channels):
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if in_channels != 3:
        old_conv1 = backbone.conv1
        new_conv1 = nn.Conv2d(in_channels, old_conv1.out_channels, kernel_size=old_conv1.kernel_size,
                               stride=old_conv1.stride, padding=old_conv1.padding, bias=False)
        with torch.no_grad():
            for i in range(in_channels):
                new_conv1.weight[:, i] = old_conv1.weight[:, i % 3]
        backbone.conv1 = new_conv1
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feat_dim

class DualFlowModel(nn.Module):
    def __init__(self, num_classes, dropout=0.4):
        super().__init__()
        self.rgb_backbone, rgb_dim = make_backbone(3)
        self.flow_backbone, flow_dim = make_backbone(4) # Dual flow: 4 channels
        self.depth_backbone, depth_dim = make_backbone(1)

        fused_dim = rgb_dim + flow_dim + depth_dim # 1536
        
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )
        
        # Frozen Backbone Discipline
        self._freeze_backbone(self.rgb_backbone)
        self._freeze_backbone(self.flow_backbone)
        self._freeze_backbone(self.depth_backbone)

    def _freeze_backbone(self, backbone):
        for name, param in backbone.named_parameters():
            if not name.startswith("layer4"):
                param.requires_grad = False

    def forward(self, rgb, flow, depth):
        rgb_feat = self.rgb_backbone(rgb)
        flow_feat = self.flow_backbone(flow)
        depth_feat = self.depth_backbone(depth)
        
        fused = torch.cat([rgb_feat, flow_feat, depth_feat], dim=1)
        return self.classifier(fused)

# --- FOCAL LOSS ---

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def compute_class_weights(df, label_map, device):
    counts = df["emotion"].value_counts()
    num_classes = len(label_map)
    weights = torch.ones(num_classes)
    total = len(df)
    for cls, idx in label_map.items():
        count = counts.get(cls, 1)
        weights[idx] = total / (num_classes * count)
    return weights.to(device)


MANIFEST_PATH = Path("CASME3_Dataset/fold_manifest.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METRICS_DIR = Path("metrics")

def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for rgb, flow, depth, y in tqdm(loader, desc="  train", leave=False):
        rgb, flow, depth, y = rgb.to(DEVICE), flow.to(DEVICE), depth.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(rgb, flow, depth)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        
    return total_loss / total, correct / total

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for rgb, flow, depth, y in tqdm(loader, desc="  val  ", leave=False):
        rgb, flow, depth, y = rgb.to(DEVICE), flow.to(DEVICE), depth.to(DEVICE), y.to(DEVICE)
        logits = model(rgb, flow, depth)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        
    uf1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    uar = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    
    return total_loss / total, acc, uf1, uar, all_labels, all_preds

def run_one_fold(fold, df, label_map, epochs, overfit_test=False):
    if overfit_test:
        tiny_df = df.sample(n=16, random_state=42).reset_index(drop=True)
        train_loader = build_dataloader(tiny_df, label_map, batch_size=8, shuffle=True, is_train=False, use_weighted_sampler=False)
        val_loader = train_loader
        lr = 1e-3
    else:
        train_df = df[df["fold"] != fold].reset_index(drop=True)
        val_df = df[df["fold"] == fold].reset_index(drop=True)
        train_loader = build_dataloader(train_df, label_map, batch_size=16, shuffle=True, is_train=True, use_weighted_sampler=False)
        val_loader = build_dataloader(val_df, label_map, batch_size=16, shuffle=False, is_train=False)
        lr = 1e-4

    model = DualFlowModel(num_classes=len(label_map), dropout=0.4).to(DEVICE)
    
    if overfit_test:
        criterion = nn.CrossEntropyLoss()
    else:
        class_weights = compute_class_weights(train_df, label_map, DEVICE)
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_uf1 = 0.0
    best_metrics = {'uf1': 0.0, 'uar': 0.0, 'acc': 0.0, 'best_epoch': 0, 'best_train_acc': 0.0}
    patience = 8
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, vl_uf1, vl_uar, val_labels, val_preds = validate(model, val_loader, criterion)
        scheduler.step()

        marker = ""
        if vl_uf1 > best_val_uf1:
            best_val_uf1 = vl_uf1
            patience_counter = 0
            best_metrics = {'uf1': vl_uf1, 'uar': vl_uar, 'acc': vl_acc, 'best_epoch': epoch, 'best_train_acc': tr_acc}
            marker = "  * best"
        else:
            patience_counter += 1
            
        print(f"Epoch {epoch:3d}/{epochs}  |  train loss={tr_loss:.3f} acc={tr_acc:.3f}  |  val loss={vl_loss:.3f} acc={vl_acc:.3f} uf1={vl_uf1:.3f}{marker}")
        
        if not overfit_test and patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
    print(f"Best val UF1: {best_val_uf1:.4f} at epoch {best_metrics['best_epoch']} (Train Acc at best: {best_metrics['best_train_acc']:.4f})")
    return best_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overfit_test", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--all_folds", action="store_true")
    args = parser.parse_args()

    seed_everything(42)
    METRICS_DIR.mkdir(exist_ok=True)
    
    df = pd.read_csv(MANIFEST_PATH)
    if 'clip_name' not in df.columns and 'clip_id' in df.columns: df['clip_name'] = df['clip_id']
    if 'fold' not in df.columns and 'fold_number' in df.columns: df['fold'] = df['fold_number']
        
    # Ensure BOTH full-face npz and offset npz exist
    df = df[df['clip_name'].apply(lambda x: (PROCESSED_DIR / f"{x}.npz").exists() and (OFFSET_DIR / f"{x}.npz").exists())].copy()
    
    df["emotion"] = df["emotion"].str.strip().str.lower()
    label_map = fit_label_map(df["emotion"])
    print(f"Total valid clips: {len(df)}")
    print("Label map:", label_map)

    if args.overfit_test:
        print("\n=== OVERFIT TEST MODE (Experiment J: Dual-Range Flow) ===\n")
        run_one_fold(0, df, label_map, epochs=args.epochs or 40, overfit_test=True)
        return

    if args.all_folds:
        print("\n=== ALL FOLDS (Experiment J: Dual-Range Flow) ===\n")
        results = {}
        for fold in sorted(df["fold"].unique()):
            print(f"\n{'='*60}\nFOLD {fold}\n{'='*60}")
            results[fold] = run_one_fold(fold, df, label_map, epochs=args.epochs or 30)
            
        uf1s = [m['uf1'] for m in results.values()]
        uars = [m['uar'] for m in results.values()]
        
        print(f"\n{'='*60}\nFINAL SUMMARY (Experiment J: Dual-Range Flow)\n{'='*60}")
        for fold, m in results.items():
            print(f"  Fold {fold}: UF1={m['uf1']:.4f}, UAR={m['uar']:.4f}")
        print("-" * 40)
        print(f"  Mean UF1: {np.mean(uf1s):.4f} ± {np.std(uf1s):.4f}")
        print(f"  Mean UAR: {np.mean(uars):.4f} ± {np.std(uars):.4f}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
