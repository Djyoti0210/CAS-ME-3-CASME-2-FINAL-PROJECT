"""
08_train_htnet.py -- Trains the lightweight HTNet-style ROI model.
Run --overfit_test FIRST, same discipline as every prior stage.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from dataset_htnet import fit_label_map, build_dataloader
from model_htnet import HTNetLite

MANIFEST_PATH = Path("manifest.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()

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
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in tqdm(loader, desc="  train", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in tqdm(loader, desc="  val  ", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total

def run_one_fold(fold, df, label_map, epochs, overfit_test=False):
    if overfit_test:
        tiny_df = df.sample(n=16, random_state=42).reset_index(drop=True)
        train_loader = build_dataloader(tiny_df, label_map, batch_size=8, shuffle=True,
                                         is_train=False, use_weighted_sampler=False)
        val_loader = train_loader
        lr = 1e-3
    else:
        train_df = df[df["fold"] != fold].reset_index(drop=True)
        val_df = df[df["fold"] == fold].reset_index(drop=True)
        print(f"train={len(train_df)}  val={len(val_df)}")
        train_loader = build_dataloader(train_df, label_map, batch_size=32, shuffle=True,
                                         is_train=True, use_weighted_sampler=False)
        val_loader = build_dataloader(val_df, label_map, batch_size=32, shuffle=False, is_train=False)
        lr = 3e-4

    model = HTNetLite(num_classes=len(label_map), embed_dim=64, num_regions=3, dropout=0.3).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}  (compare to ResNet18's ~11.7M)")

    if overfit_test:
        criterion = nn.CrossEntropyLoss()
    else:
        class_weights = compute_class_weights(train_df, label_map, DEVICE)
        criterion = FocalLoss(alpha=class_weights, gamma=1.5)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_epoch = 0
    patience = 12
    patience_counter = 0
    ckpt_path = Path(f"best_htnet_fold{fold}.pt") if not overfit_test else None

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = validate(model, val_loader, criterion)
        scheduler.step()
        marker = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_epoch = epoch
            patience_counter = 0
            if ckpt_path:
                torch.save(model.state_dict(), ckpt_path)
            marker = "  * saved"
        else:
            patience_counter += 1
        print(f"Epoch {epoch:3d}/{epochs}  |  train loss={tr_loss:.3f} acc={tr_acc:.3f}  |  "
              f"val loss={vl_loss:.3f} acc={vl_acc:.3f}{marker}")
        if not overfit_test and patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (best val acc {best_val_acc:.3f} at epoch {best_epoch})")
            break

    print(f"Best val acc: {best_val_acc:.3f} at epoch {best_epoch}")
    return best_val_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overfit_test", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all_folds", action="store_true")
    args = parser.parse_args()

    seed_everything(42)
    df = pd.read_csv(MANIFEST_PATH)
    df["emotion"] = df["emotion"].str.strip().str.lower()

    roi_dir = Path("processed_roi")
    df["roi_exists"] = df["clip_name"].apply(lambda c: (roi_dir / f"{c}.npz").exists())
    n_before = len(df)
    df = df[df["roi_exists"]].reset_index(drop=True)
    print(f"Clips with ROI data: {len(df)} / {n_before}")

    label_map = fit_label_map(df["emotion"])
    print("Label map:", label_map)
    print("Device:", DEVICE)

    if args.overfit_test:
        print("\n=== OVERFIT TEST MODE (HTNet-Lite) ===\n")
        run_one_fold(0, df, label_map, epochs=args.epochs or 100, overfit_test=True)
        return

    if args.all_folds:
        print("\n=== ALL FOLDS (HTNet-Lite) ===\n")
        results = {}
        for fold in sorted(df["fold"].unique()):
            print(f"\n{'='*60}\nFOLD {fold}\n{'='*60}")
            acc = run_one_fold(fold, df, label_map, epochs=args.epochs or 60)
            results[f"fold_{fold}"] = acc
        accs = list(results.values())
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
        print(f"  Mean: {np.mean(accs):.4f}  Std: {np.std(accs):.4f}")
    else:
        print(f"\n=== FOLD {args.fold} (HTNet-Lite) ===\n")
        run_one_fold(args.fold, df, label_map, epochs=args.epochs or 60)

if __name__ == "__main__":
    main()