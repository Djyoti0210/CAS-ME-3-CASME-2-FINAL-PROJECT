"""
09_compute_uf1_uar.py -- Computes UF1 (unweighted F1 = macro F1) and UAR
(unweighted average recall = macro recall) for both the two-stream and
three-stream(+focal loss) models, aggregated across all 5 folds. These are
the actual metrics used in CAS(ME)3 published literature -- NOT raw accuracy.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score, classification_report

MANIFEST_PATH = Path("manifest.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def build_twostream_model():
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

def evaluate_model(model_type, df, label_map, class_names):
    num_classes = len(label_map)
    all_preds, all_labels = [], []

    if model_type == "twostream":
        TwoStreamModel = build_twostream_model()
    else:
        from model import ThreeStreamModel

    for fold in sorted(df["fold"].unique()):
        if model_type == "twostream":
            ckpt_path = Path(f"best_twostream_fold{fold}.pt")
            model = TwoStreamModel(num_classes=num_classes).to(DEVICE)
        else:
            ckpt_path = Path(f"best_threestream_fold{fold}.pt")
            model = ThreeStreamModel(num_classes=num_classes).to(DEVICE)

        if not ckpt_path.exists():
            print(f"WARNING: {ckpt_path} not found, skipping fold {fold}")
            continue

        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        model.eval()

        val_df = df[df["fold"] == fold].reset_index(drop=True)
        with torch.no_grad():
            for _, row in val_df.iterrows():
                rgb, flow, depth = load_sample(row["clip_name"])
                rgb, flow, depth = rgb.unsqueeze(0).to(DEVICE), flow.unsqueeze(0).to(DEVICE), depth.unsqueeze(0).to(DEVICE)

                if model_type == "twostream":
                    logits = model(rgb, flow)
                else:
                    logits = model(rgb, flow, depth)

                pred = logits.argmax(1).item()
                label = label_map[row["emotion"]]
                all_preds.append(pred)
                all_labels.append(label)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = np.mean(all_preds == all_labels)
    uf1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    uar = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    return accuracy, uf1, uar, all_preds, all_labels

def main():
    df = pd.read_csv(MANIFEST_PATH)
    df["emotion"] = df["emotion"].str.strip().str.lower()
    label_map = fit_label_map(df["emotion"])
    class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]

    print("=" * 70)
    print("Evaluating TWO-STREAM model (RGB + Flow)")
    print("=" * 70)
    acc2, uf1_2, uar_2, preds2, labels2 = evaluate_model("twostream", df, label_map, class_names)
    print(f"Accuracy: {acc2:.4f}   UF1: {uf1_2:.4f}   UAR: {uar_2:.4f}")

    print("\n" + "=" * 70)
    print("Evaluating THREE-STREAM + Focal Loss model (RGB + Flow + Depth)")
    print("=" * 70)
    acc3, uf1_3, uar_3, preds3, labels3 = evaluate_model("threestream", df, label_map, class_names)
    print(f"Accuracy: {acc3:.4f}   UF1: {uf1_3:.4f}   UAR: {uar_3:.4f}")

    print("\n" + "=" * 70)
    print("COMPARISON TO PUBLISHED CAS(ME)3 BENCHMARKS (LOSO protocol, same task)")
    print("=" * 70)
    published = [
        ("AlexNet",   0.30, 0.30),
        ("FeatRef",   0.35, 0.34),
        ("STSTNet",   0.38, 0.38),
        ("RCN-A",     0.39, 0.39),
        ("MEAN",      0.39, 0.40),
        ("SFAMNet",   0.45, 0.48),
        ("ME-TST",    0.48, 0.49),
        ("SOTA (2025 paper)", 0.56, 0.55),
    ]
    print(f"{'Method':<25}{'UF1':>8}{'UAR':>8}")
    print("-" * 41)
    for name, uf1, uar in published:
        print(f"{name:<25}{uf1:>8.2f}{uar:>8.2f}")
    print("-" * 41)
    print(f"{'OUR two-stream':<25}{uf1_2:>8.2f}{uar_2:>8.2f}")
    print(f"{'OUR three-stream+focal':<25}{uf1_3:>8.2f}{uar_3:>8.2f}")

    print("\n" + "=" * 70)
    print("Full classification report -- BEST model (three-stream+focal)")
    print("=" * 70)
    print(classification_report(labels3, preds3, target_names=class_names, zero_division=0))

    from sklearn.metrics import confusion_matrix
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)
    print("Rows: True Label | Columns: Predicted Label")
    print(f"Order: {class_names}")
    print("-" * 70)
    cm = confusion_matrix(labels3, preds3, labels=range(len(class_names)))
    print(cm)

if __name__ == "__main__":
    main()
