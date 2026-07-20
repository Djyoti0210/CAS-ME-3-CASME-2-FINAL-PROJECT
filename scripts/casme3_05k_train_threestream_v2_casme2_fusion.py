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

# --- GOD-TIER MODEL DEFINITION ---

def make_convnext_backbone():
    backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    feat_dim = backbone.classifier[2].in_features
    # Remove the final linear layer so it acts as a feature extractor
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

class TransformerFusionModel(nn.Module):
    def __init__(self, num_classes, dropout=0.4):
        super().__init__()
        # 1. RGB uses the CASME2 ConvNeXt-Tiny Champion architecture
        self.rgb_backbone, rgb_dim = make_convnext_backbone()
        # Flow & Depth use standard ResNet18
        self.flow_backbone, flow_dim = make_resnet_backbone(2)
        self.depth_backbone, depth_dim = make_resnet_backbone(1)

        # 2. Projection layers to make all streams the same dimension (d_model=512)
        self.d_model = 512
        self.rgb_proj = nn.Sequential(nn.Linear(rgb_dim, self.d_model), nn.GELU())
        self.flow_proj = nn.Sequential(nn.Linear(flow_dim, self.d_model), nn.GELU())
        self.depth_proj = nn.Sequential(nn.Linear(depth_dim, self.d_model), nn.GELU())
        
        # 3. God-Tier Transformer Cross-Attention Block
        # Allows streams to contextualize each other before classification
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=8, 
            dim_feedforward=1024, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 4. Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model * 3, 256),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )
        
        # Initial Freezing: Freeze RGB entirely, leave layer4 of flow/depth unfrozen
        self._freeze_all(self.rgb_backbone)
        self._freeze_resnet_backbone(self.flow_backbone)
        self._freeze_resnet_backbone(self.depth_backbone)

    def _freeze_all(self, backbone):
        for param in backbone.parameters():
            param.requires_grad = False

    def unfreeze_casme2_rgb(self):
        # Unfreeze layer 7 of ConvNeXt to allow synergistic fine-tuning in Stage 2
        for name, param in self.rgb_backbone.named_parameters():
            if "features.7" in name or "classifier" in name:
                param.requires_grad = True

    def _freeze_resnet_backbone(self, backbone):
        for name, param in backbone.named_parameters():
            if not name.startswith("layer4"):
                param.requires_grad = False

    def forward(self, rgb, flow, depth):
        # Extract features
        r = self.rgb_backbone(rgb)
        f = self.flow_backbone(flow)
        d = self.depth_backbone(depth)
        
        # Project to d_model and add sequence dimension [Batch, 1, d_model]
        r_proj = self.rgb_proj(r).unsqueeze(1)
        f_proj = self.flow_proj(f).unsqueeze(1)
        d_proj = self.depth_proj(d).unsqueeze(1)
        
        # Combine into sequence [Batch, 3, d_model]
        tokens = torch.cat([r_proj, f_proj, d_proj], dim=1)
        
        # Transformer Cross-Attention (The Magic Happens Here)
        attended_tokens = self.transformer(tokens)
        
        # Flatten and classify [Batch, 1536]
        fused = attended_tokens.reshape(attended_tokens.size(0), -1)
        return self.classifier(fused)

# --- LABEL SMOOTHED FOCAL LOSS ---

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = label_smoothing

    def forward(self, logits, targets):
        # Label smoothing prevents overconfidence on the majority class
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
    plt.savefig(METRICS_DIR / f"fold_{fold}_fusion_learning_curves.png", dpi=150)
    plt.close()

def plot_confusion_matrix(cm, class_names, fold):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(METRICS_DIR / f"fold_{fold}_fusion_cm.png", dpi=150)
    plt.close()

def train_epoch(model, loader, criterion, optimizer, accum_steps=4):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    
    optimizer.zero_grad()
    
    for i, (rgb, flow, depth, y) in enumerate(tqdm(loader, desc="  train", leave=False)):
        rgb, flow, depth, y = rgb.to(DEVICE), flow.to(DEVICE), depth.to(DEVICE), y.to(DEVICE)
        
        logits = model(rgb, flow, depth)
        
        # Gradient Accumulation: Simulate Batch Size of 64 (16 * 4) for ultra-stable Focal Loss
        loss = criterion(logits, y) / accum_steps
        loss.backward()
        
        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accum_steps * y.size(0)
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

def run_one_fold(fold, df, label_map, epochs):
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)
    
    train_loader = build_dataloader(train_df, label_map, batch_size=16, shuffle=True, is_train=True, use_weighted_sampler=False)
    val_loader = build_dataloader(val_df, label_map, batch_size=16, shuffle=False, is_train=False)

    model = TransformerFusionModel(num_classes=len(label_map), dropout=0.4).to(DEVICE)
    
    # NOTE: As per documentation and verification logs, this checkpoint comes from a 
    # 5-fold CLIP-LEVEL cross-validation run on the CASME2 static pool, NOT a strict 
    # subject-independent LOSO setup, as the static pool lacked reliable per-clip subject boundaries.
    # The weights inside this .pt file are from a final 100% all-data pass (60 epochs).
    pretrained_path = "casme2_5fold_backbone_champion.pt"
    if Path(pretrained_path).exists():
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Enforce safe unpickling using weights_only=True
            sd = torch.load(pretrained_path, map_location=DEVICE, weights_only=True)
            
        # Strip out the classifier layer of ConvNeXt because dimensions don't match our fusion head
        sd = {k: v for k, v in sd.items() if not k.startswith("classifier.2")}
        model.rgb_backbone.load_state_dict(sd, strict=False)
        print(f"Loaded CASME2 (5-fold Clip-Level) pretrained RGB backbone from {pretrained_path}")
    else:
        print(f"WARNING: Could not find {pretrained_path}")
    
    class_weights = compute_class_weights(train_df, label_map, DEVICE)
    # Applied Label Smoothing to Focal Loss to prevent overconfidence
    criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.1)
        
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_uf1 = 0.0
    best_metrics = {'uf1': 0.0, 'uar': 0.0, 'acc': 0.0}
    best_epoch = 0
    patience = 12
    patience_counter = 0
    ckpt_path = Path(f"casme3_best_threestream_fusion_fold{fold}.pt")

    history = {'train_loss': [], 'val_loss': [], 'val_uf1': [], 'val_acc': []}
    metrics_log = []

    for epoch in range(1, epochs + 1):
        # Two-Stage Gradual Unfreezing Logic
        if epoch == 11:
            print("\n  >>> ENTERING STAGE 2: Unfreezing CASME2 RGB backbone (ConvNeXt Features.7) for synergistic fine-tuning!\n")
            model.unfreeze_casme2_rgb()

        # accum_steps=4 gives an effective batch size of 64
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, accum_steps=4)
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
        # Strictly monitor UF1 for saving and early stopping
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
            
            marker = "  * saved"
        else:
            patience_counter += 1
            
        print(f"Epoch {epoch:2d}/{epochs}  |  train loss={tr_loss:.3f} acc={tr_acc:.3f}  |  val loss={vl_loss:.3f} acc={vl_acc:.3f} uf1={vl_uf1:.3f}{marker}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    pd.DataFrame(metrics_log).to_csv(METRICS_DIR / f"fold_{fold}_fusion_metrics.csv", index=False)
    plot_learning_curves(history, fold)
        
    print(f"Best val UF1: {best_val_uf1:.4f} at epoch {best_epoch}")
    return best_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
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
        print("\n=== ALL FOLDS CAS(ME)3 GOD-TIER FUSION (Transformer Cross-Attention) ===\n")
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
