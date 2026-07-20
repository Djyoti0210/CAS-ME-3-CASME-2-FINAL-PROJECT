"""
06_analyze_errors.py -- Loads best checkpoint per fold, runs predictions on
that fold's val set, aggregates confusion matrix and per-class metrics across
ALL folds. This tells you exactly which classes are dragging accuracy down.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report

from dataset import fit_label_map, build_dataloader
from model import ThreeStreamModel

MANIFEST_PATH = Path("manifest.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    df = pd.read_csv(MANIFEST_PATH)
    df["emotion"] = df["emotion"].str.strip().str.lower()
    label_map = fit_label_map(df["emotion"])
    class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    num_classes = len(label_map)

    all_preds, all_labels = [], []

    for fold in sorted(df["fold"].unique()):
        ckpt_path = Path(f"best_threestream_fold{fold}.pt")
        if not ckpt_path.exists():
            print(f"WARNING: {ckpt_path} not found, skipping fold {fold}")
            continue

        val_df = df[df["fold"] == fold].reset_index(drop=True)
        val_loader = build_dataloader(val_df, label_map, batch_size=16, shuffle=False, is_train=False)

        model = ThreeStreamModel(num_classes=num_classes, dropout=0.5).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.eval()

        with torch.no_grad():
            for rgb, flow, depth, y in val_loader:
                rgb, flow, depth = rgb.to(DEVICE), flow.to(DEVICE), depth.to(DEVICE)
                logits = model(rgb, flow, depth)
                preds = logits.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("=" * 60)
    print("AGGREGATED CONFUSION MATRIX (rows=true, cols=predicted)")
    print("=" * 60)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df.to_string())

    print("\n" + "=" * 60)
    print("PER-CLASS REPORT")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    print("\n" + "=" * 60)
    print("PER-CLASS SAMPLE COUNT IN VAL (aggregated across folds)")
    print("=" * 60)
    unique, counts = np.unique(all_labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {class_names[u]}: {c}")

if __name__ == "__main__":
    main()
