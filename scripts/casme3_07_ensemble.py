"""
07_ensemble.py -- Averages softmax predictions from the two-stream (RGB+Flow)
and three-stream (RGB+Flow+Depth) checkpoints for each fold, then reports
per-fold and aggregate accuracy + full confusion matrix / classification report.
Requires: best_twostream_fold{i}.pt and best_threestream_fold{i}.pt to exist
for each fold (from earlier training runs).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report

# Two-stream model/dataset (RGB+Flow)
import importlib
import sys

MANIFEST_PATH = Path("manifest.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_twostream_components():
    """Two-stream model needs its own dataset.py/model.py definitions.
    Since dataset.py/model.py on disk currently hold the THREE-stream version,
    we define the two-stream architecture inline here to avoid needing the
    old files back."""
    import torch.nn as nn
    import torchvision.models as models

    def make_rgb_backbone():
        backbone = models.resnet18(weights=None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feat_dim

    def make_flow_backbone():
        backbone = models.resnet18(weights=None)
        old_conv1 = backbone.conv1
        new_conv1 = nn.Conv2d(2, old_conv1.out_channels, kernel_size=old_conv1.kernel_size,
                               stride=old_conv1.stride, padding=old_conv1.padding, bias=False)
        backbone.conv1 = new_conv1
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feat_dim

    class TwoStreamModel(nn.Module):
        def __init__(self, num_classes, dropout=0.5):
            super().__init__()
            self.rgb_backbone, rgb_dim = make_rgb_backbone()
            self.flow_backbone, flow_dim = make_flow_backbone()
            fused_dim = rgb_dim + flow_dim
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout), nn.Linear(fused_dim, 256), nn.ReLU(inplace=True),
                nn.Dropout(p=dropout), nn.Linear(256, num_classes)
            )
        def forward(self, rgb, flow):
            rgb_feat = self.rgb_backbone(rgb)
            flow_feat = self.flow_backbone(flow)
            fused = torch.cat([rgb_feat, flow_feat], dim=1)
            return self.classifier(fused)

    return TwoStreamModel


def fit_label_map(emotions):
    unique = sorted(emotions.unique())
    return {e: i for i, e in enumerate(unique)}


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PROCESSED_DIR = Path("processed")


def load_sample(clip_name):
    data = np.load(PROCESSED_DIR / f"{clip_name}.npz")
    rgb = data["rgb"].copy().astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    flow = data["flow"].copy()
    flow = np.clip(flow / 15.0, -1.0, 1.0)
    depth = data["depth"].copy().astype(np.float32) / 255.0
    depth = depth[..., None]

    rgb_t = torch.from_numpy(rgb).float().permute(2, 0, 1)
    flow_t = torch.from_numpy(flow).float().permute(2, 0, 1)
    depth_t = torch.from_numpy(depth).float().permute(2, 0, 1)
    return rgb_t, flow_t, depth_t


def main():
    df = pd.read_csv(MANIFEST_PATH)
    df["emotion"] = df["emotion"].str.strip().str.lower()
    label_map = fit_label_map(df["emotion"])
    class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    num_classes = len(label_map)

    from model import ThreeStreamModel  # current three-stream model.py on disk
    TwoStreamModel = load_twostream_components()

    all_preds, all_labels = [], []
    fold_accs = {}

    for fold in sorted(df["fold"].unique()):
        two_ckpt = Path(f"best_twostream_fold{fold}.pt")
        three_ckpt = Path(f"best_threestream_fold{fold}.pt")

        if not two_ckpt.exists() or not three_ckpt.exists():
            print(f"WARNING: missing checkpoint(s) for fold {fold}, skipping. "
                  f"two_exists={two_ckpt.exists()} three_exists={three_ckpt.exists()}")
            continue

        two_model = TwoStreamModel(num_classes=num_classes).to(DEVICE)
        two_model.load_state_dict(torch.load(two_ckpt, map_location=DEVICE, weights_only=True))
        two_model.eval()

        three_model = ThreeStreamModel(num_classes=num_classes).to(DEVICE)
        three_model.load_state_dict(torch.load(three_ckpt, map_location=DEVICE, weights_only=True))
        three_model.eval()

        val_df = df[df["fold"] == fold].reset_index(drop=True)
        fold_preds, fold_labels = [], []

        with torch.no_grad():
            for _, row in val_df.iterrows():
                rgb, flow, depth = load_sample(row["clip_name"])
                rgb, flow, depth = rgb.unsqueeze(0).to(DEVICE), flow.unsqueeze(0).to(DEVICE), depth.unsqueeze(0).to(DEVICE)

                logits_two = two_model(rgb, flow)
                logits_three = three_model(rgb, flow, depth)

                probs_two = F.softmax(logits_two, dim=1)
                probs_three = F.softmax(logits_three, dim=1)
                avg_probs = (probs_two + probs_three) / 2.0

                pred = avg_probs.argmax(1).item()
                label = label_map[row["emotion"]]

                fold_preds.append(pred)
                fold_labels.append(label)

        fold_acc = np.mean(np.array(fold_preds) == np.array(fold_labels))
        fold_accs[f"fold_{fold}"] = fold_acc
        print(f"Fold {fold}: ensemble val acc = {fold_acc:.4f}")

        all_preds.extend(fold_preds)
        all_labels.extend(fold_labels)

    print("\n" + "=" * 60)
    print("ENSEMBLE SUMMARY")
    print("=" * 60)
    for k, v in fold_accs.items():
        print(f"  {k}: {v:.4f}")
    accs = list(fold_accs.values())
    print(f"  Mean: {np.mean(accs):.4f}  Std: {np.std(accs):.4f}")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("\n" + "=" * 60)
    print("AGGREGATED CONFUSION MATRIX (ensemble)")
    print("=" * 60)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    print(pd.DataFrame(cm, index=class_names, columns=class_names).to_string())

    print("\n" + "=" * 60)
    print("PER-CLASS REPORT (ensemble)")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))


if __name__ == "__main__":
    main()
