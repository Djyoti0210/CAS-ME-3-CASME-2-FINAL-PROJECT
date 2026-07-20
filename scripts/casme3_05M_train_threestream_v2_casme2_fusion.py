import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, recall_score, confusion_matrix, accuracy_score

from casme3_dataset import fit_label_map, build_dataloader
import torchvision.models as models

# --- LATE FUSION MODEL DEFINITION ---

def make_convnext_backbone():
    backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    feat_dim = backbone.classifier[2].in_features
    backbone.classifier[2] = nn.Identity()
    return backbone, feat_dim

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
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feat_dim

class LateFusionAuxModel(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        # 1. Backbones
        self.rgb_backbone, rgb_dim = make_convnext_backbone()
        self.flow_backbone, flow_dim = make_resnet_backbone(2)
        self.depth_backbone, depth_dim = make_resnet_backbone(1)

        # 2. Independent Classifiers (Brains) for each stream
        self.rgb_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(rgb_dim, num_classes)
        )
        self.flow_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(flow_dim, num_classes)
        )
        self.depth_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(depth_dim, num_classes)
        )
        
        # 3. Learnable Logit Ensembling weights
        # Initializes equally: 33% RGB, 33% Flow, 33% Depth
        self.fusion_weights = nn.Parameter(torch.ones(3))
        
        # Initialize Freezing Discipline
        self._freeze_resnet_backbone(self.flow_backbone)
        self._freeze_resnet_backbone(self.depth_backbone)
        self._unfreeze_casme2_rgb_partial()

    def _freeze_resnet_backbone(self, backbone):
        for name, param in backbone.named_parameters():
            if not name.startswith("layer4"):
                param.requires_grad = False

    def _unfreeze_casme2_rgb_partial(self):
        for name, param in self.rgb_backbone.named_parameters():
            if "features.7" in name or "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, rgb, flow, depth):
        r = self.rgb_backbone(rgb)
        f = self.flow_backbone(flow)
        d = self.depth_backbone(depth)
        
        # Each stream thinks independently
        logits_rgb = self.rgb_head(r)
        logits_flow = self.flow_head(f)
        logits_depth = self.depth_head(d)
        
        # Dynamic Ensembling (Softmax guarantees they sum to 100%)
        w = F.softmax(self.fusion_weights, dim=0)
        logits_fused = (w[0] * logits_rgb) + (w[1] * logits_flow) + (w[2] * logits_depth)
        
        if self.training:
            # During training, return all logits for Auxiliary Supervision
            return logits_fused, logits_rgb, logits_flow, logits_depth
            
        # During validation/testing, just return the fused final prediction
        return logits_fused

# --- FOCAL LOSS ---

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none', label_smoothing=self.smoothing)
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


MANIFEST_PATH = Path("manifest_casme3_767.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METRICS_DIR = Path("metrics")

def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def plot_learning_curves(history, fold):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(epochs, history['train_loss'], label='Train Loss (Total)')
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
    plt.savefig(METRICS_DIR / f"fold_{fold}_latefusion_learning_curves.png", dpi=150)
    plt.close()

def plot_confusion_matrix(cm, class_names, fold):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(METRICS_DIR / f"fold_{fold}_latefusion_cm.png", dpi=150)
    plt.close()

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for rgb, flow, depth, y in tqdm(loader, desc="  train", leave=False):
        rgb, flow, depth, y = rgb.to(DEVICE), flow.to(DEVICE), depth.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        
        # Forward pass yields 4 sets of predictions!
        logits_fused, logits_rgb, logits_flow, logits_depth = model(rgb, flow, depth)
        
        loss_fused = criterion(logits_fused, y)
        loss_rgb = criterion(logits_rgb, y)
        loss_flow = criterion(logits_flow, y)
        loss_depth = criterion(logits_depth, y)
        
        # Deep Auxiliary Supervision: 
        # Force every stream to learn independently so nobody gets lazy!
        loss = loss_fused + (0.5 * loss_rgb) + (0.5 * loss_flow) + (0.5 * loss_depth)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * y.size(0)
        preds = logits_fused.argmax(1)
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
    for rgb, flow, depth, y in tqdm(loader, desc="  val  ", leave=False):
        rgb, flow, depth, y = rgb.to(DEVICE), flow.to(DEVICE), depth.to(DEVICE), y.to(DEVICE)
        
        # In eval mode, it only returns the fused logits
        logits_fused = model(rgb, flow, depth)
        
        loss = criterion(logits_fused, y)
        total_loss += loss.item() * y.size(0)
        preds = logits_fused.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        
    uf1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    uar = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    
    return total_loss / total, acc, uf1, uar, all_labels, all_preds

def run_one_fold(fold, df, label_map, epochs):
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)
    
    train_loader = build_dataloader(train_df, label_map, batch_size=16, shuffle=True, is_train=True, use_weighted_sampler=False)
    val_loader = build_dataloader(val_df, label_map, batch_size=16, shuffle=False, is_train=False)

    model = LateFusionAuxModel(num_classes=len(label_map), dropout=0.5).to(DEVICE)
    
    # NOTE: Checkpoint from 5-fold CLIP-LEVEL cross-validation run on the CASME2 static pool.
    pretrained_path = "casme2_5fold_backbone_champion.pt"
    if Path(pretrained_path).exists():
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sd = torch.load(pretrained_path, map_location=DEVICE, weights_only=True)
            
        sd = {k: v for k, v in sd.items() if not k.startswith("classifier.2")}
        model.rgb_backbone.load_state_dict(sd, strict=False)
        print(f"Loaded CASME2 pretrained RGB backbone from {pretrained_path}")
    else:
        print(f"WARNING: Could not find {pretrained_path}")
    
    class_weights = compute_class_weights(train_df, label_map, DEVICE)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.2)
        
    # Differential Learning Rates (ConvNeXt uses 1e-5, everything else 1e-4)
    rgb_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "rgb_backbone" in name:
            rgb_params.append(param)
        else:
            other_params.append(param)
            
    optimizer = optim.AdamW([
        {'params': rgb_params, 'lr': 1e-5},
        {'params': other_params, 'lr': 1e-4}
    ], weight_decay=1e-3)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_uf1 = 0.0
    best_metrics = {'uf1': 0.0, 'uar': 0.0, 'acc': 0.0}
    best_epoch = 0
    patience = 12
    patience_counter = 0
    ckpt_path = Path(f"casme3_best_threestream_latefusion_fold{fold}.pt")

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
        # Strictly monitor UF1
        if vl_uf1 > best_val_uf1:
            best_val_uf1 = vl_uf1
            best_epoch = epoch
            patience_counter = 0
            best_metrics = {'uf1': vl_uf1, 'uar': vl_uar, 'acc': vl_acc}
            
            if ckpt_path:
                torch.save(model.state_dict(), ckpt_path)
            
            class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
            cm = confusion_matrix(val_labels, val_preds, labels=range(len(label_map)))
            plot_confusion_matrix(cm, class_names, fold)
            
            # Print current fusion weights
            w = F.softmax(model.fusion_weights, dim=0).detach().cpu().numpy()
            marker = f"  * saved (Weights: RGB={w[0]:.2f}, Flow={w[1]:.2f}, Depth={w[2]:.2f})"
        else:
            patience_counter += 1
            
        print(f"Epoch {epoch:2d}/{epochs}  |  train loss={tr_loss:.3f} acc={tr_acc:.3f}  |  val loss={vl_loss:.3f} acc={vl_acc:.3f} uf1={vl_uf1:.3f}{marker}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    pd.DataFrame(metrics_log).to_csv(METRICS_DIR / f"fold_{fold}_latefusion_metrics.csv", index=False)
    plot_learning_curves(history, fold)
        
    print(f"Best val UF1: {best_val_uf1:.4f} at epoch {best_epoch}")
    return best_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
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
        
    PROCESSED_DIR = Path("processed")
    df = df[df['clip_name'].apply(lambda x: (PROCESSED_DIR / f"{x}.npz").exists())].copy()
    
    df["emotion"] = df["emotion"].str.strip().str.lower()
    label_map = fit_label_map(df["emotion"])
    print("Label map:", label_map)
    print("Device:", DEVICE)

    if args.all_folds:
        print("\n=== ALL FOLDS LATE FUSION (Deep Auxiliary Supervision) ===\n")
        results = {}
        for fold in sorted(df["fold"].unique()):
            print(f"\n{'='*60}\nFOLD {fold}\n{'='*60}")
            metrics = run_one_fold(fold, df, label_map, epochs=args.epochs)
            results[fold] = metrics
            
        uf1s = [m['uf1'] for m in results.values()]
        uars = [m['uar'] for m in results.values()]
        
        print(f"\n{'='*60}\nFINAL FUSION SUMMARY\n{'='*60}")
        for fold, m in results.items():
            print(f"  Fold {fold}: UF1={m['uf1']:.4f}, UAR={m['uar']:.4f}")
        print("-" * 40)
        print(f"  Mean UF1: {np.mean(uf1s):.4f} ± {np.std(uf1s):.4f}")
        print(f"  Mean UAR: {np.mean(uars):.4f} ± {np.std(uars):.4f}")
        print(f"{'='*60}\n")
    else:
        print(f"\n=== FOLD {args.fold} FUSION ===\n")
        run_one_fold(args.fold, df, label_map, epochs=args.epochs)

if __name__ == "__main__":
    main()
