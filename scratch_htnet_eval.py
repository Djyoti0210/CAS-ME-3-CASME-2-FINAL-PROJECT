import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import sys

sys.path.append("scripts")
from model_htnet import HTNetLite
from dataset_htnet import ROIDataset
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MANIFEST_PATH = Path("manifest.csv")

def fit_label_map(emotions):
    unique = sorted(emotions.unique())
    return {e: i for i, e in enumerate(unique)}

def main():
    df = pd.read_csv(MANIFEST_PATH)
    df["emotion"] = df["emotion"].str.strip().str.lower()
    roi_dir = Path("processed_roi")
    df["roi_exists"] = df["clip_name"].apply(lambda c: (roi_dir / f"{c}.npz").exists())
    df = df[df["roi_exists"]].reset_index(drop=True)

    label_map = fit_label_map(df["emotion"])
    class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    
    all_preds = []
    all_labels = []

    for fold in sorted(df["fold"].unique()):
        ckpt = Path(f"best_htnet_fold{fold}.pt")
        if not ckpt.exists():
            continue
        
        # Initialize HTNetLite with 3 regions as the user modified it
        model = HTNetLite(num_classes=len(label_map), embed_dim=64, num_regions=3, dropout=0.0).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        model.eval()

        val_df = df[df["fold"] == fold].reset_index(drop=True)
        val_dataset = ROIDataset(val_df, label_map, is_train=False)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                logits = model(x)
                preds = logits.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())

    print("HTNET LITE RECALL REPORT")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    print("\nRAW CONFUSION MATRIX:")
    print("Rows: True Labels | Columns: Predicted Labels")
    print("Class Order:", class_names)
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    print(cm)

if __name__ == "__main__":
    main()
