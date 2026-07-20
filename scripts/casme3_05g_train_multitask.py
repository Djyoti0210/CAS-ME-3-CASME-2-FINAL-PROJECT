import argparse
import os
import re
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, recall_score, accuracy_score
import torchvision.models as models

# --- DATASET & AU PARSING ---

PROCESSED_DIR = Path("processed")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def fit_label_map(emotions):
    unique = sorted(emotions.unique())
    return {e: i for i, e in enumerate(unique)}

def get_top_k_aus(df, k=10):
    all_aus = []
    for idx, row in df.iterrows():
        au_str = str(row.get('Action Unit', row.get('AU', row.get('au', '')))) # Handle different column names
        aus = re.findall(r'\d+', au_str)
        all_aus.extend(aus)
    counts = Counter(all_aus)
    top_k = [x[0] for x in counts.most_common(k)]
    return {au: i for i, au in enumerate(top_k)}

class MultiTaskDataset(Dataset):
    def __init__(self, df, emo_label_map, au_label_map, is_train=False):
        self.df = df.reset_index(drop=True)
        self.emo_label_map = emo_label_map
        self.au_label_map = au_label_map
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clip_name = row['clip_name']
        
        npz_path = PROCESSED_DIR / f"{clip_name}.npz"
        data = np.load(npz_path)
        
        rgb = data["rgb"].copy().astype(np.float32) / 255.0
        flow = data["flow"].copy()
        flow = np.clip(flow / 15.0, -1.0, 1.0)
        depth = data["depth"].copy().astype(np.float32) / 255.0
        depth = depth[..., None]

        if self.is_train:
            import random
            if random.random() > 0.5:
                rgb = np.flip(rgb, axis=1).copy()
                flow = np.flip(flow, axis=1).copy()
                flow[..., 0] = -flow[..., 0]
                depth = np.flip(depth, axis=1).copy()

        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD

        rgb_t = torch.from_numpy(rgb).float().permute(2, 0, 1)
        flow_t = torch.from_numpy(flow).float().permute(2, 0, 1)
        depth_t = torch.from_numpy(depth).float().permute(2, 0, 1)

        emo_label = self.emo_label_map[str(row["emotion"]).strip().lower()]
        
        y_au = torch.zeros(len(self.au_label_map), dtype=torch.float32)
        au_str = str(row.get('Action Unit', row.get('AU', row.get('au', ''))))
        aus = re.findall(r'\d+', au_str)
        for au in aus:
            if au in self.au_label_map:
                y_au[self.au_label_map[au]] = 1.0

        return rgb_t, flow_t, depth_t, torch.tensor(emo_label, dtype=torch.long), y_au

def build_dataloader(df, emo_label_map, au_label_map, batch_size=16, shuffle=True, is_train=False, use_weighted_sampler=True):
    dataset = MultiTaskDataset(df, emo_label_map, au_label_map, is_train=is_train)
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
            if in_channels == 1:
                new_conv1.weight[:, 0] = old_conv1.weight.mean(dim=1)
            elif in_channels == 2:
                new_conv1.weight[:, 0] = old_conv1.weight[:, 0]
                new_conv1.weight[:, 1] = old_conv1.weight[:, 1]
        backbone.conv1 = new_conv1
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feat_dim

class MultiTaskModel(nn.Module):
    def __init__(self, num_classes, num_aus, dropout=0.4):
        super().__init__()
        self.rgb_backbone, rgb_dim = make_backbone(3)
        self.flow_backbone, flow_dim = make_backbone(2)
        self.depth_backbone, depth_dim = make_backbone(1)

        fused_dim = rgb_dim + flow_dim + depth_dim # 1536
        
        # Primary Emotion Head
        self.emotion_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )
        
        # Secondary AU Head
        self.au_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_aus)
        )
        
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
        
        logits_emo = self.emotion_head(fused)
        logits_au = self.au_head(fused)
        return logits_emo, logits_au

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

def train_epoch(model, loader, criterion_emo, criterion_au, optimizer, au_weight=0.3):
    model.train()
    total_emo_loss, total_au_loss, correct, total = 0.0, 0.0, 0, 0
    for rgb, flow, depth, y_emo, y_au in tqdm(loader, desc="  train", leave=False):
        rgb, flow, depth = rgb.to(DEVICE), flow.to(DEVICE), depth.to(DEVICE)
        y_emo, y_au = y_emo.to(DEVICE), y_au.to(DEVICE)
        
        optimizer.zero_grad()
        logits_emo, logits_au = model(rgb, flow, depth)
        
        loss_emo = criterion_emo(logits_emo, y_emo)
        loss_au = criterion_au(logits_au, y_au)
        
        loss = loss_emo + au_weight * loss_au
        loss.backward()
        optimizer.step()
        
        total_emo_loss += loss_emo.item() * y_emo.size(0)
        total_au_loss += loss_au.item() * y_emo.size(0)
        
        preds = logits_emo.argmax(1)
        correct += (preds == y_emo).sum().item()
        total += y_emo.size(0)
        
    return total_emo_loss / total, total_au_loss / total, correct / total

@torch.no_grad()
def validate(model, loader, criterion_emo, criterion_au):
    model.eval()
    total_emo_loss, total_au_loss, correct, total = 0.0, 0.0, 0, 0
    all_preds, all_labels = [], []
    for rgb, flow, depth, y_emo, y_au in tqdm(loader, desc="  val  ", leave=False):
        rgb, flow, depth = rgb.to(DEVICE), flow.to(DEVICE), depth.to(DEVICE)
        y_emo, y_au = y_emo.to(DEVICE), y_au.to(DEVICE)
        
        logits_emo, logits_au = model(rgb, flow, depth)
        
        loss_emo = criterion_emo(logits_emo, y_emo)
        loss_au = criterion_au(logits_au, y_au)
        
        total_emo_loss += loss_emo.item() * y_emo.size(0)
        total_au_loss += loss_au.item() * y_emo.size(0)
        
        preds = logits_emo.argmax(1)
        correct += (preds == y_emo).sum().item()
        total += y_emo.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_emo.cpu().numpy())
        
    uf1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    uar = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    
    return total_emo_loss / total, total_au_loss / total, acc, uf1, uar

def run_one_fold(fold, df, emo_label_map, au_label_map, epochs, overfit_test=False):
    if overfit_test:
        tiny_df = df.sample(n=16, random_state=42).reset_index(drop=True)
        train_loader = build_dataloader(tiny_df, emo_label_map, au_label_map, batch_size=8, shuffle=True, is_train=False, use_weighted_sampler=False)
        val_loader = train_loader
        lr = 1e-3
    else:
        train_df = df[df["fold"] != fold].reset_index(drop=True)
        val_df = df[df["fold"] == fold].reset_index(drop=True)
        train_loader = build_dataloader(train_df, emo_label_map, au_label_map, batch_size=16, shuffle=True, is_train=True, use_weighted_sampler=False)
        val_loader = build_dataloader(val_df, emo_label_map, au_label_map, batch_size=16, shuffle=False, is_train=False)
        lr = 1e-4

    model = MultiTaskModel(num_classes=len(emo_label_map), num_aus=len(au_label_map), dropout=0.4).to(DEVICE)
    
    if overfit_test:
        criterion_emo = nn.CrossEntropyLoss()
    else:
        class_weights = compute_class_weights(train_df, emo_label_map, DEVICE)
        criterion_emo = FocalLoss(alpha=class_weights, gamma=2.0)
        
    criterion_au = nn.BCEWithLogitsLoss()
        
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_uf1 = 0.0
    best_metrics = {'uf1': 0.0, 'uar': 0.0, 'acc': 0.0, 'best_epoch': 0, 'best_train_acc': 0.0}
    patience = 8
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        tr_emo_loss, tr_au_loss, tr_acc = train_epoch(model, train_loader, criterion_emo, criterion_au, optimizer, au_weight=0.3)
        vl_emo_loss, vl_au_loss, vl_acc, vl_uf1, vl_uar = validate(model, val_loader, criterion_emo, criterion_au)
        scheduler.step()

        marker = ""
        if vl_uf1 > best_val_uf1:
            best_val_uf1 = vl_uf1
            patience_counter = 0
            best_metrics = {'uf1': vl_uf1, 'uar': vl_uar, 'acc': vl_acc, 'best_epoch': epoch, 'best_train_acc': tr_acc}
            marker = "  * best"
        else:
            patience_counter += 1
            
        print(f"Ep {epoch:2d}/{epochs} | Tr EmoL={tr_emo_loss:.3f} AuL={tr_au_loss:.3f} Acc={tr_acc:.3f} | Vl EmoL={vl_emo_loss:.3f} AuL={vl_au_loss:.3f} UF1={vl_uf1:.3f}{marker}")
        
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
        
    df = df[df['clip_name'].apply(lambda x: (PROCESSED_DIR / f"{x}.npz").exists())].copy()
    
    df["emotion"] = df["emotion"].str.strip().str.lower()
    emo_label_map = fit_label_map(df["emotion"])
    au_label_map = get_top_k_aus(df, k=10)
    
    print(f"Total valid clips: {len(df)}")
    print("Emotion Label map:", emo_label_map)
    print("Top-10 AU Label map:", au_label_map)

    if args.overfit_test:
        print("\n=== OVERFIT TEST MODE (Experiment K: AU Multi-Task) ===\n")
        run_one_fold(0, df, emo_label_map, au_label_map, epochs=args.epochs or 40, overfit_test=True)
        return

    if args.all_folds:
        print("\n=== ALL FOLDS (Experiment K: AU Multi-Task) ===\n")
        results = {}
        for fold in sorted(df["fold"].unique()):
            print(f"\n{'='*60}\nFOLD {fold}\n{'='*60}")
            results[fold] = run_one_fold(fold, df, emo_label_map, au_label_map, epochs=args.epochs or 30)
            
        uf1s = [m['uf1'] for m in results.values()]
        uars = [m['uar'] for m in results.values()]
        
        print(f"\n{'='*60}\nFINAL SUMMARY (Experiment K: AU Multi-Task)\n{'='*60}")
        for fold, m in results.items():
            print(f"  Fold {fold}: UF1={m['uf1']:.4f}, UAR={m['uar']:.4f}")
        print("-" * 40)
        print(f"  Mean UF1: {np.mean(uf1s):.4f} ± {np.std(uf1s):.4f}")
        print(f"  Mean UAR: {np.mean(uars):.4f} ± {np.std(uars):.4f}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
