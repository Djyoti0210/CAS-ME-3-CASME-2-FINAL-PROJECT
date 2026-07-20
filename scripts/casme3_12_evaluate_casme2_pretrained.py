"""
12_evaluate_casme2_pretrained.py -- Computes UF1 and UAR
for the three-stream models, comparing the ImageNet-initialized champion
to the CASME2-pretrained RGB champion.
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

def evaluate_model(model_type, df, label_map, class_names):
    num_classes = len(label_map)
    all_preds, all_labels = [], []
    fold_uf1s, fold_uars = [], []

    for fold in sorted(df["fold"].unique()):
        if model_type == "threestream":
            ckpt_path = Path(f"best_threestream_fold{fold}.pt")
            from model import ThreeStreamModel
            model = ThreeStreamModel(num_classes=num_classes).to(DEVICE)
        elif model_type == "threestream_v2":
            ckpt_path = Path(f"best_threestream_v2_fold{fold}.pt")
            import importlib.util
            import sys
            spec = importlib.util.spec_from_file_location("mod_v2", f"scripts/05b_train_threestream_v2.py")
            mod_v2 = importlib.util.module_from_spec(spec)
            sys.modules["mod_v2"] = mod_v2
            # Add scripts to sys.path so it can import dataset
            if "scripts" not in sys.path: sys.path.insert(0, "scripts")
            spec.loader.exec_module(mod_v2)
            ThreeStreamModelV2 = mod_v2.ThreeStreamModelV2
            model = ThreeStreamModelV2(num_classes=num_classes).to(DEVICE)
        else:
            ckpt_path = Path(f"best_threestream_casme2_fold{fold}.pt")
            from model import ThreeStreamModel
            model = ThreeStreamModel(num_classes=num_classes).to(DEVICE)

        if not ckpt_path.exists():
            print(f"WARNING: {ckpt_path} not found, skipping fold {fold}")
            continue

        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        model.eval()

        val_df = df[df["fold"] == fold].reset_index(drop=True)
        fold_preds, fold_labels = [], []
        with torch.no_grad():
            for _, row in val_df.iterrows():
                rgb, flow, depth = load_sample(row["clip_name"])
                rgb, flow, depth = rgb.unsqueeze(0).to(DEVICE), flow.unsqueeze(0).to(DEVICE), depth.unsqueeze(0).to(DEVICE)

                logits = model(rgb, flow, depth)

                pred = logits.argmax(1).item()
                label = label_map[row["emotion"]]
                fold_preds.append(pred)
                fold_labels.append(label)

        fold_preds = np.array(fold_preds)
        fold_labels = np.array(fold_labels)
        
        all_preds.extend(fold_preds)
        all_labels.extend(fold_labels)
        
        f_uf1 = f1_score(fold_labels, fold_preds, average="macro", zero_division=0)
        f_uar = recall_score(fold_labels, fold_preds, average="macro", zero_division=0)
        fold_uf1s.append(f_uf1)
        fold_uars.append(f_uar)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = np.mean(all_preds == all_labels)
    uf1_global = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    uar_global = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    
    uf1_mean = np.mean(fold_uf1s)
    uar_mean = np.mean(fold_uars)
    uf1_std = np.std(fold_uf1s)
    uar_std = np.std(fold_uars)

    return accuracy, uf1_global, uar_global, uf1_mean, uar_mean, uf1_std, uar_std, all_preds, all_labels, fold_uf1s, fold_uars

def main():
    df = pd.read_csv(MANIFEST_PATH)
    df["emotion"] = df["emotion"].str.strip().str.lower()
    label_map = fit_label_map(df["emotion"])
    class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]

    print("=" * 70)
    print("Evaluating BASELINE CHAMPION (ImageNet RGB init)")
    print("=" * 70)
    acc2, uf1_2, uar_2, uf1_m2, uar_m2, uf1_s2, uar_s2, preds2, labels2, f_uf1s2, f_uars2 = evaluate_model("threestream", df, label_map, class_names)
    print(f"Accuracy: {acc2:.4f}   Global UF1: {uf1_2:.4f}   Global UAR: {uar_2:.4f}")
    print("\nPer-Fold UF1/UAR:")
    for i, (f1, rec) in enumerate(zip(f_uf1s2, f_uars2)):
        print(f"  Fold {i}: UF1 = {f1:.4f}, UAR = {rec:.4f}")
    print(f"Mean UF1: {uf1_m2:.4f} ± {uf1_s2:.4f}")
    print(f"Mean UAR: {uar_m2:.4f} ± {uar_s2:.4f}")
    print("\nClassification Report (BASELINE):")
    print(classification_report(labels2, preds2, target_names=class_names, zero_division=0))


    print("\n" + "=" * 70)
    print("Evaluating BASELINE CHAMPION V2 (best_threestream_v2_foldX.pt)")
    print("=" * 70)
    acc_v2, uf1_v2, uar_v2, uf1_mv2, uar_mv2, uf1_sv2, uar_sv2, preds_v2, labels_v2, f_uf1s_v2, f_uars_v2 = evaluate_model("threestream_v2", df, label_map, class_names)
    print(f"Accuracy: {acc_v2:.4f}   Global UF1: {uf1_v2:.4f}   Global UAR: {uar_v2:.4f}")
    print("\nPer-Fold UF1/UAR:")
    for i, (f1, rec) in enumerate(zip(f_uf1s_v2, f_uars_v2)):
        print(f"  Fold {i}: UF1 = {f1:.4f}, UAR = {rec:.4f}")
    print(f"Mean UF1: {uf1_mv2:.4f} ± {uf1_sv2:.4f}")
    print(f"Mean UAR: {uar_mv2:.4f} ± {uar_sv2:.4f}")
    print("\nClassification Report (BASELINE V2):")
    print(classification_report(labels_v2, preds_v2, target_names=class_names, zero_division=0))

    print("\n" + "=" * 70)
    print("Evaluating CASME2-PRETRAINED CHAMPION (CASME2 RGB init)")
    print("=" * 70)
    acc3, uf1_3, uar_3, uf1_m3, uar_m3, uf1_s3, uar_s3, preds3, labels3, f_uf1s3, f_uars3 = evaluate_model("threestream_casme2", df, label_map, class_names)
    print(f"Accuracy: {acc3:.4f}   Global UF1: {uf1_3:.4f}   Global UAR: {uar_3:.4f}")
    print("\nPer-Fold UF1/UAR:")
    for i, (f1, rec) in enumerate(zip(f_uf1s3, f_uars3)):
        print(f"  Fold {i}: UF1 = {f1:.4f}, UAR = {rec:.4f}")
    print(f"Mean UF1: {uf1_m3:.4f} ± {uf1_s3:.4f}")
    print(f"Mean UAR: {uar_m3:.4f} ± {uar_s3:.4f}")
    print("\nClassification Report (CASME2 PRETRAINED):")
    print(classification_report(labels3, preds3, target_names=class_names, zero_division=0))

if __name__ == "__main__":
    main()
