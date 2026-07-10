"""
10_tta_ensemble_calibrated.py -- Combines three free improvements using
ALREADY-TRAINED checkpoints (no retraining):
  1. Test-Time Augmentation (TTA): average predictions over original + flipped input
  2. Weighted ensemble: three-stream+focal weighted higher than two-stream
  3. Class-prior calibration: adjust logits to counteract majority-class bias
Reports UF1/UAR before and after each addition so you can see what helps.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score, classification_report

MANIFEST_PATH = Path("manifest.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PROCESSED_DIR = Path("processed")

def fit_label_map(emotions):
    unique = sorted(emotions.unique())
    return {e: i for i, e in enumerate(unique)}

def load_sample(clip_name, flip=False):
    data = np.load(PROCESSED_DIR / f"{clip_name}.npz")
    rgb = data["rgb"].copy().astype(np.float32) / 255.0
    flow = data["flow"].copy()
    flow = np.clip(flow / 15.0, -1.0, 1.0)
    depth = data["depth"].copy().astype(np.float32) / 255.0
    depth = depth[..., None]

    if flip:
        rgb = np.flip(rgb, axis=1).copy()
        flow = np.flip(flow, axis=1).copy()
        flow[..., 0] = -flow[..., 0]
        depth = np.flip(depth, axis=1).copy()

    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD

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

def get_tta_probs(model, model_type, rgb, flow, depth, rgb_f, flow_f, depth_f):
    with torch.no_grad():
        if model_type == "twostream":
            logits1 = model(rgb, flow)
            logits2 = model(rgb_f, flow_f)
        else:
            logits1 = model(rgb, flow, depth)
            logits2 = model(rgb_f, flow_f, depth_f)
        probs = (F.softmax(logits1, dim=1) + F.softmax(logits2, dim=1)) / 2.0
    return probs

def main():
    df = pd.read_csv(MANIFEST_PATH)
    df["emotion"] = df["emotion"].str.strip().str.lower()
    label_map = fit_label_map(df["emotion"])
    class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    num_classes = len(label_map)

    from model import ThreeStreamModel
    TwoStreamModel = build_twostream_model()

    class_counts = df["emotion"].value_counts()
    total = len(df)
    class_prior = np.array([class_counts.get(c, 1) / total for c in
                             [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]])

    all_probs_two, all_probs_three, all_labels = [], [], []

    for fold in sorted(df["fold"].unique()):
        two_ckpt = Path(f"best_twostream_fold{fold}.pt")
        three_ckpt = Path(f"best_threestream_fold{fold}.pt")
        if not two_ckpt.exists() or not three_ckpt.exists():
            print(f"WARNING: missing checkpoint for fold {fold}, skipping")
            continue

        two_model = TwoStreamModel(num_classes=num_classes).to(DEVICE)
        two_model.load_state_dict(torch.load(two_ckpt, map_location=DEVICE, weights_only=True))
        two_model.eval()

        three_model = ThreeStreamModel(num_classes=num_classes).to(DEVICE)
        three_model.load_state_dict(torch.load(three_ckpt, map_location=DEVICE, weights_only=True))
        three_model.eval()

        val_df = df[df["fold"] == fold].reset_index(drop=True)

        for _, row in val_df.iterrows():
            rgb, flow, depth = load_sample(row["clip_name"], flip=False)
            rgb_f, flow_f, depth_f = load_sample(row["clip_name"], flip=True)
            rgb, flow, depth = rgb.unsqueeze(0).to(DEVICE), flow.unsqueeze(0).to(DEVICE), depth.unsqueeze(0).to(DEVICE)
            rgb_f, flow_f, depth_f = rgb_f.unsqueeze(0).to(DEVICE), flow_f.unsqueeze(0).to(DEVICE), depth_f.unsqueeze(0).to(DEVICE)

            probs_two = get_tta_probs(two_model, "twostream", rgb, flow, depth, rgb_f, flow_f, depth_f)
            probs_three = get_tta_probs(three_model, "threestream", rgb, flow, depth, rgb_f, flow_f, depth_f)

            all_probs_two.append(probs_two.cpu().numpy()[0])
            all_probs_three.append(probs_three.cpu().numpy()[0])
            all_labels.append(label_map[row["emotion"]])

    all_probs_two = np.array(all_probs_two)
    all_probs_three = np.array(all_probs_three)
    all_labels = np.array(all_labels)

    def report(name, probs):
        preds = probs.argmax(1)
        uf1 = f1_score(all_labels, preds, average="macro", zero_division=0)
        uar = recall_score(all_labels, preds, average="macro", zero_division=0)
        acc = np.mean(preds == all_labels)
        print(f"{name:<45}  Acc={acc:.4f}  UF1={uf1:.4f}  UAR={uar:.4f}")
        return preds, uf1, uar

    print("=" * 80)
    print("STEP-BY-STEP IMPROVEMENTS (all using existing checkpoints, no retraining)")
    print("=" * 80)

    report("1. Two-stream + TTA", all_probs_two)
    report("2. Three-stream+focal + TTA", all_probs_three)

    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        combined = w * all_probs_two + (1 - w) * all_probs_three
        report(f"3. Weighted ensemble (two={w:.1f}, three={1-w:.1f})", combined)

    best_w = 0.3
    combined_best = best_w * all_probs_two + (1 - best_w) * all_probs_three
    calibrated = combined_best / (class_prior[None, :] ** 0.5)
    calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
    preds_cal, uf1_cal, uar_cal = report("4. Weighted ensemble + prior calibration", calibrated)

    print("\n" + "=" * 80)
    print("FINAL CLASSIFICATION REPORT (best combination)")
    print("=" * 80)
    print(classification_report(all_labels, preds_cal, target_names=class_names, zero_division=0))

    from sklearn.metrics import confusion_matrix
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)
    print("Rows: True Label | Columns: Predicted Label")
    print(f"Order: {class_names}")
    print("-" * 80)
    cm = confusion_matrix(all_labels, preds_cal, labels=range(len(class_names)))
    print(cm)

if __name__ == "__main__":
    main()
