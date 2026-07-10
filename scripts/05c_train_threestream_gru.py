import argparse
import os
import random
import cv2
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, recall_score, confusion_matrix, accuracy_score

from dataset import fit_label_map
import torchvision.models as models

# --- CONSTANTS ---
MANIFEST_PATH = Path("CASME3_Dataset/fold_manifest.csv")
PROCESSED_DIR = Path("processed")
ROI_DIR = Path("CASME3_Dataset/extracted_rois_v3")
METRICS_DIR = Path("metrics")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# --- DATASET DEFINITION ---
class ThreeStreamGRUDataset(Dataset):
    def __init__(self, df, label_map, is_train=False):
        self.df = df.reset_index(drop=True)
        self.label_map = label_map
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def _process_flow_seq(self, f_seq):
        # f_seq shape: (3, H, W, 2)
        resized = []
        for i in range(f_seq.shape[0]):
            frame = cv2.resize(f_seq[i], (28, 28))
            resized.append(frame)
        resized_seq = np.stack(resized, axis=0) # (3, 28, 28, 2)
        return resized_seq.reshape(3, -1) # (3, 1568)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clip_name = row['clip_name']
        
        # Load RGB and Depth
        npz_path = PROCESSED_DIR / f"{clip_name}.npz"
        data = np.load(npz_path)
        rgb = data["rgb"].copy().astype(np.float32) / 255.0
        depth = data["depth"].copy().astype(np.float32) / 255.0
        depth = depth[..., None]

        # Load flow sequences
        roi_clip_dir = ROI_DIR / clip_name
        left = np.load(roi_clip_dir / "left_eye.npz")["flow_seq"].copy()
        right = np.load(roi_clip_dir / "right_eye.npz")["flow_seq"].copy()
        mouth = np.load(roi_clip_dir / "mouth.npz")["flow_seq"].copy()

        # Data augmentation
        if self.is_train and random.random() > 0.5:
            rgb = np.flip(rgb, axis=1).copy()
            depth = np.flip(depth, axis=1).copy()
            
            # Swap left and right eye because of horizontal flip
            left, right = right, left
            
            # Flip spatially along W axis (axis=2 for shape 3, H, W, 2)
            left = np.flip(left, axis=2).copy()
            left[..., 0] = -left[..., 0] # Negate X flow
            
            right = np.flip(right, axis=2).copy()
            right[..., 0] = -right[..., 0]
            
            mouth = np.flip(mouth, axis=2).copy()
            mouth[..., 0] = -mouth[..., 0]

        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        rgb_t = torch.from_numpy(rgb).float().permute(2, 0, 1)
        depth_t = torch.from_numpy(depth).float().permute(2, 0, 1)

        left_t = torch.from_numpy(self._process_flow_seq(left)).float()
        right_t = torch.from_numpy(self._process_flow_seq(right)).float()
        mouth_t = torch.from_numpy(self._process_flow_seq(mouth)).float()

        label = self.label_map[str(row["emotion"]).strip().lower()]
        return rgb_t, depth_t, left_t, right_t, mouth_t, torch.tensor(label, dtype=torch.long)

def build_dataloader(df, label_map, batch_size=16, shuffle=True, is_train=False):
    dataset = ThreeStreamGRUDataset(df, label_map, is_train=is_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

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
        backbone.conv1 = new_conv1
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feat_dim

class ThreeStreamGRUModel(nn.Module):
    def __init__(self, num_classes, dropout=0.4):
        super().__init__()
        self.rgb_backbone, rgb_dim = make_backbone(3)
        self.depth_backbone, depth_dim = make_backbone(1)

        # Frozen Backbone Discipline: Only layer4 and the classifier are trainable
        self._freeze_backbone(self.rgb_backbone)
        self._freeze_backbone(self.depth_backbone)

        # GRU modules for Flow sequences
        # input_size = 28 * 28 * 2 = 1568
        self.left_gru = nn.GRU(input_size=1568, hidden_size=128, num_layers=1, batch_first=True)
        self.right_gru = nn.GRU(input_size=1568, hidden_size=128, num_layers=1, batch_first=True)
        self.mouth_gru = nn.GRU(input_size=1568, hidden_size=128, num_layers=1, batch_first=True)
        
        flow_dim = 128 * 3 # 384

        fused_dim = rgb_dim + flow_dim + depth_dim # 512 + 384 + 512 = 1408
        
        # Keep small fusion head pattern
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )

    def _freeze_backbone(self, backbone):
        for name, param in backbone.named_parameters():
            if not name.startswith("layer4"):
                param.requires_grad = False

    def forward(self, rgb, depth, left_seq, right_seq, mouth_seq):
        rgb_feat = self.rgb_backbone(rgb)
        depth_feat = self.depth_backbone(depth)
        
        _, left_h = self.left_gru(left_seq)   # (1, B, 128)
        _, right_h = self.right_gru(right_seq)
        _, mouth_h = self.mouth_gru(mouth_seq)
        
        left_feat = left_h.squeeze(0)   # (B, 128)
        right_feat = right_h.squeeze(0) # (B, 128)
        mouth_feat = mouth_h.squeeze(0) # (B, 128)
        
        fused = torch.cat([rgb_feat, left_feat, right_feat, mouth_feat, depth_feat], dim=1)
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

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def plot_learning_curves(history, fold):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(epochs, history['train_loss'], label='Train Loss')
    ax1.plot(epochs, history['val_loss'], label='Val Loss')
    ax1.set_title(f'Loss Curve - Fold {fold}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    if 'val_uf1' in history:
        ax2.plot(epochs, history['val_uf1'], label='Val UF1', color='green')
        ax2.set_title(f'Macro-F1 (UF1) - Fold {fold}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('UF1')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(METRICS_DIR / f"exp_f_fold_{fold}_learning_curves.png", dpi=150)
    plt.close()

def plot_confusion_matrix(cm, class_names, fold):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(METRICS_DIR / f"exp_f_fold_{fold}_cm.png", dpi=150)
    plt.close()

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for rgb, depth, left, right, mouth, y in tqdm(loader, desc="  train", leave=False):
        rgb, depth, left, right, mouth, y = rgb.to(DEVICE), depth.to(DEVICE), left.to(DEVICE), right.to(DEVICE), mouth.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(rgb, depth, left, right, mouth)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        
    return total_loss / total, correct / total

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for rgb, depth, left, right, mouth, y in tqdm(loader, desc="  val  ", leave=False):
        rgb, depth, left, right, mouth, y = rgb.to(DEVICE), depth.to(DEVICE), left.to(DEVICE), right.to(DEVICE), mouth.to(DEVICE), y.to(DEVICE)
        logits = model(rgb, depth, left, right, mouth)
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
        train_loader = build_dataloader(tiny_df, label_map, batch_size=8, shuffle=True, is_train=False)
        val_loader = train_loader
        lr = 1e-3
    else:
        train_df = df[df["fold"] != fold].reset_index(drop=True)
        val_df = df[df["fold"] == fold].reset_index(drop=True)
        train_loader = build_dataloader(train_df, label_map, batch_size=16, shuffle=True, is_train=True)
        val_loader = build_dataloader(val_df, label_map, batch_size=16, shuffle=False, is_train=False)
        lr = 1e-4

    model = ThreeStreamGRUModel(num_classes=len(label_map), dropout=0.4).to(DEVICE)
    
    if overfit_test:
        criterion = nn.CrossEntropyLoss()
    else:
        class_weights = compute_class_weights(train_df, label_map, DEVICE)
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3) # wd=1e-3
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_uf1 = 0.0
    best_metrics = {'uf1': 0.0, 'uar': 0.0, 'acc': 0.0, 'best_epoch': 0, 'best_train_acc': 0.0}
    patience = 8
    patience_counter = 0
    ckpt_path = Path(f"best_threestream_gru_fold{fold}.pt") if not overfit_test else None

    history = {'train_loss': [], 'val_loss': [], 'val_uf1': [], 'val_acc': []}
    metrics_log = []

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, vl_uf1, vl_uar, val_labels, val_preds = validate(model, val_loader, criterion)
        scheduler.step()
        
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['val_uf1'].append(vl_uf1)
        history['val_acc'].append(vl_acc)
        
        metrics_log.append({
            'epoch': epoch, 'train_loss': tr_loss, 'train_acc': tr_acc,
            'val_loss': vl_loss, 'val_acc': vl_acc, 'val_uf1': vl_uf1, 'val_uar': vl_uar
        })

        marker = ""
        if vl_uf1 > best_val_uf1:
            best_val_uf1 = vl_uf1
            patience_counter = 0
            best_metrics = {'uf1': vl_uf1, 'uar': vl_uar, 'acc': vl_acc, 'best_epoch': epoch, 'best_train_acc': tr_acc}
            
            if ckpt_path:
                torch.save(model.state_dict(), ckpt_path)
            
            class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
            cm = confusion_matrix(val_labels, val_preds, labels=range(len(label_map)))
            if not overfit_test:
                plot_confusion_matrix(cm, class_names, fold)
            
            marker = "  * saved"
        else:
            patience_counter += 1
            
        print(f"Epoch {epoch:3d}/{epochs}  |  train loss={tr_loss:.3f} acc={tr_acc:.3f}  |  val loss={vl_loss:.3f} acc={vl_acc:.3f} uf1={vl_uf1:.3f}{marker}")
        
        if not overfit_test and patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if not overfit_test:
        pd.DataFrame(metrics_log).to_csv(METRICS_DIR / f"exp_f_fold_{fold}_metrics.csv", index=False)
        plot_learning_curves(history, fold)
        
    print(f"Best val UF1: {best_val_uf1:.4f} at epoch {best_metrics['best_epoch']} (Train Acc at best: {best_metrics['best_train_acc']:.4f})")
    return best_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overfit_test", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all_folds", action="store_true")
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
        
    # Strict filtering: require processed NPZ AND the 3 ROI NPZs
    valid_clips = []
    for c in df['clip_name']:
        has_processed = (PROCESSED_DIR / f"{c}.npz").exists()
        roi_dir = ROI_DIR / c
        has_rois = (roi_dir / "left_eye.npz").exists() and (roi_dir / "right_eye.npz").exists() and (roi_dir / "mouth.npz").exists()
        if has_processed and has_rois:
            valid_clips.append(c)
            
    df = df[df['clip_name'].isin(valid_clips)].copy()
    
    df["emotion"] = df["emotion"].str.strip().str.lower()
    label_map = fit_label_map(df["emotion"])
    print("Label map:", label_map)
    print(f"Total valid clips: {len(df)}")
    print("Device:", DEVICE)

    if args.overfit_test:
        print("\n=== OVERFIT TEST MODE (Experiment F: GRU Temporal Flow) ===\n")
        run_one_fold(0, df, label_map, epochs=args.epochs or 40, overfit_test=True)
        return

    if args.all_folds:
        print("\n=== ALL FOLDS (Experiment F: GRU Temporal Flow) ===\n")
        results = {}
        for fold in sorted(df["fold"].unique()):
            print(f"\n{'='*60}\nFOLD {fold}\n{'='*60}")
            metrics = run_one_fold(fold, df, label_map, epochs=args.epochs or 30)
            results[fold] = metrics
            
        uf1s = [m['uf1'] for m in results.values()]
        uars = [m['uar'] for m in results.values()]
        
        print(f"\n{'='*60}\nFINAL SUMMARY (Experiment F)\n{'='*60}")
        for fold, m in results.items():
            print(f"  Fold {fold}: UF1={m['uf1']:.4f}, UAR={m['uar']:.4f}, BestEpoch={m['best_epoch']}, TrainAcc@Best={m['best_train_acc']:.4f}")
        print("-" * 40)
        print(f"  Mean UF1: {np.mean(uf1s):.4f} ± {np.std(uf1s):.4f}")
        print(f"  Mean UAR: {np.mean(uars):.4f} ± {np.std(uars):.4f}")
        print(f"{'='*60}\n")
    else:
        print(f"\n=== FOLD {args.fold} (Experiment F: GRU Temporal Flow) ===\n")
        run_one_fold(args.fold, df, label_map, epochs=args.epochs or 30)

if __name__ == "__main__":
    main()
