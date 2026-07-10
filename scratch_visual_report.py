import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
import torch
from torch.utils.data import DataLoader

sys.path.append("scripts")
from dataset_temporal import TemporalDataset
from model_threestream import ThreeStreamNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MANIFEST_PATH = Path("manifest.csv")
ARTIFACTS_DIR = Path(r"C:\Users\sanch\.gemini\antigravity-ide\brain\947c132a-f711-4eae-8af9-db1d39be7e50\artifacts")

def fit_label_map(emotions):
    unique = sorted(emotions.unique())
    return {e: i for i, e in enumerate(unique)}

def generate_report():
    df = pd.read_csv(MANIFEST_PATH)
    df["emotion"] = df["emotion"].str.strip().str.lower()
    
    # Check what files exist
    rgb_dir = Path("processed/rgb")
    if not rgb_dir.exists():
        rgb_dir = Path("processed")
    
    df["rgb_exists"] = df["clip_name"].apply(lambda c: (rgb_dir / f"{c}.npz").exists())
    df = df[df["rgb_exists"]].reset_index(drop=True)
    
    label_map = fit_label_map(df["emotion"])
    class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    
    all_preds = []
    all_labels = []
    
    # Gather three-stream predictions
    for fold in sorted(df["fold"].unique()):
        ckpt = Path(f"best_threestream_fold{fold}.pt")
        if not ckpt.exists():
            continue
            
        model = ThreeStreamNetwork(num_classes=len(class_names), dropout=0.0).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        model.eval()
        
        val_df = df[df["fold"] == fold].reset_index(drop=True)
        # Using same dataset logic as 09_compute_uf1_uar.py:
        val_dataset = TemporalDataset(val_df, label_map, is_train=False)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        
        with torch.no_grad():
            for batch in val_loader:
                # 09_compute_uf1_uar.py uses batch of size 1 directly
                rgb, flow, mask, y = batch
                rgb = rgb.to(DEVICE)
                flow = flow.to(DEVICE)
                
                # ThreeStream uses rgb and flow
                logits = model(rgb, flow)
                preds = logits.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())
                
    # If no predictions (maybe model not found), fallback to loading 09 script's output? 
    # Actually we can just run the inferences. It takes 1-2 minutes.
    
    # Generate Confusion Matrix image
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Three-Stream + Focal Loss Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "threestream_cm.png")
    plt.close()
    
    # Generate Class Balance vs Recall image
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    recalls = [report[c]['recall'] * 100 for c in class_names]
    supports = [report[c]['support'] for c in class_names]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax2 = ax1.twinx()
    ax1.bar(class_names, supports, color='lightgray', alpha=0.7, label='Support (Count)')
    ax2.plot(class_names, recalls, color='red', marker='o', linewidth=2, markersize=8, label='Recall (%)')
    
    ax1.set_xlabel('Emotion Class')
    ax1.set_ylabel('Number of Samples (Support)')
    ax2.set_ylabel('Recall Percentage (%)')
    ax1.set_title('Class Imbalance vs Model Recall (Three-Stream + Focal)')
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "threestream_recall_vs_support.png")
    plt.close()

if __name__ == "__main__":
    generate_report()
