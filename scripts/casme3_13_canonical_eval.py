"""
13_canonical_eval.py -- CANONICAL EVALUATION SCRIPT (single source of truth)

METRIC DEFINITION (LOCKED, do not change):
    Mean UF1 = average of 5 independently-computed per-fold macro-F1 scores.
               This is the academically standard method for LOSO reporting as
               used in the CAS(ME)^3 benchmark literature. Every number
               reported going forward MUST come from this exact script and
               this exact calculation.

    Mean UAR = average of 5 independently-computed per-fold macro-recall scores.

    Global UF1 / Global UAR are reported for completeness ONLY and must NOT
    be used as the primary reported metric.

Models evaluated (all checkpoints evaluated identically):
  A. best_threestream_foldX.pt        -- Original Three-Stream + Focal Loss champion
  B. best_threestream_v2_foldX.pt     -- ThreeStreamModelV2 (Exp C / eval_output.txt champion)
  C. best_threestream_casme2_foldX.pt -- CASME2 RGB-pretrained Three-Stream

Usage:
    python scripts/13_canonical_eval.py
"""

import sys
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score, recall_score, classification_report
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--manifest", default="manifest_casme3_767.csv",
                    help="Path to manifest CSV (clip_name, emotion, fold columns required)")
args, _ = parser.parse_known_args()
MANIFEST_PATH = Path(args.manifest)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PROCESSED_DIR = Path("processed")

# Ensure scripts/ is importable
if str(Path("scripts").resolve()) not in sys.path:
    sys.path.insert(0, "scripts")

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def fit_label_map(emotions):
    unique = sorted(emotions.unique())
    return {e: i for i, e in enumerate(unique)}


def load_sample(clip_name):
    data  = np.load(PROCESSED_DIR / f"{clip_name}.npz")
    rgb   = data["rgb"].copy().astype(np.float32) / 255.0
    rgb   = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    flow  = data["flow"].copy()
    flow  = np.clip(flow / 15.0, -1.0, 1.0)
    depth = data["depth"].copy().astype(np.float32) / 255.0
    depth = depth[..., None]
    rgb_t   = torch.from_numpy(rgb).float().permute(2, 0, 1)
    flow_t  = torch.from_numpy(flow).float().permute(2, 0, 1)
    depth_t = torch.from_numpy(depth).float().permute(2, 0, 1)
    return rgb_t, flow_t, depth_t


# ---------------------------------------------------------------------------
# Model loader helpers
# ---------------------------------------------------------------------------
def load_threestream_model(num_classes, ckpt_path):
    """Load the original ThreeStreamModel (casme3_model.py)."""
    from casme3_model import ThreeStreamModel
    model = ThreeStreamModel(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(
        torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    )
    return model


def load_threestream_v2_model(num_classes, ckpt_path):
    """Load ThreeStreamModelV2 from 05b_train_threestream_v2.py."""
    spec = importlib.util.spec_from_file_location(
        "mod_v2", "scripts/casme3_05b_train_threestream_v2.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mod_v2"] = mod
    spec.loader.exec_module(mod)
    ThreeStreamModelV2 = mod.ThreeStreamModelV2
    model = ThreeStreamModelV2(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(
        torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    )
    return model


# ---------------------------------------------------------------------------
# CANONICAL evaluation function
# ---------------------------------------------------------------------------
def evaluate_canonical(label, ckpt_pattern, model_loader_fn, df, label_map, class_names):
    """
    Evaluate one checkpoint family using the canonical Mean UF1 method.

    ckpt_pattern : e.g. "best_threestream_fold{fold}.pt"
    model_loader_fn : callable(num_classes, ckpt_path) -> model

    Returns dict with all metrics.
    """
    num_classes  = len(label_map)
    fold_uf1s    = []
    fold_uars    = []
    all_preds    = []
    all_labels   = []

    for fold in sorted(df["fold"].unique()):
        ckpt_path = Path(ckpt_pattern.format(fold=fold))
        if not ckpt_path.exists():
            print(f"  [WARNING] {ckpt_path} not found — skipping fold {fold}")
            continue

        model = model_loader_fn(num_classes, ckpt_path)
        model.eval()

        val_df = df[df["fold"] == fold].reset_index(drop=True)
        fold_preds  = []
        fold_labels = []

        skipped = 0
        with torch.no_grad():
            for _, row in val_df.iterrows():
                npz_path = PROCESSED_DIR / f"{row['clip_name']}.npz"
                if not npz_path.exists():
                    skipped += 1
                    continue
                rgb, flow, depth = load_sample(row["clip_name"])
                rgb   = rgb.unsqueeze(0).to(DEVICE)
                flow  = flow.unsqueeze(0).to(DEVICE)
                depth = depth.unsqueeze(0).to(DEVICE)
                logits = model(rgb, flow, depth)
                pred   = logits.argmax(1).item()
                label_ = label_map[row["emotion"]]
                fold_preds.append(pred)
                fold_labels.append(label_)
        if skipped:
            print(f"  [INFO] Skipped {skipped} clip(s) with missing .npz in fold {fold}")

        fold_preds  = np.array(fold_preds)
        fold_labels = np.array(fold_labels)

        # Per-fold metrics (the canonical computation unit)
        fold_uf1 = f1_score(fold_labels, fold_preds, average="macro", zero_division=0)
        fold_uar = recall_score(fold_labels, fold_preds, average="macro", zero_division=0)
        fold_uf1s.append(fold_uf1)
        fold_uars.append(fold_uar)

        all_preds.extend(fold_preds.tolist())
        all_labels.extend(fold_labels.tolist())

        print(f"  Fold {fold}: UF1 = {fold_uf1:.4f}  UAR = {fold_uar:.4f}  "
              f"(n={len(fold_preds)})")

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # CANONICAL metrics
    mean_uf1 = float(np.mean(fold_uf1s))
    mean_uar = float(np.mean(fold_uars))
    std_uf1  = float(np.std(fold_uf1s))
    std_uar  = float(np.std(fold_uars))

    # Global (pooled) metrics -- reported for completeness, NOT canonical
    global_acc = float(np.mean(all_preds == all_labels))
    global_uf1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    global_uar = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "label":      label,
        "mean_uf1":   mean_uf1,
        "std_uf1":    std_uf1,
        "mean_uar":   mean_uar,
        "std_uar":    std_uar,
        "global_acc": global_acc,
        "global_uf1": global_uf1,
        "global_uar": global_uar,
        "fold_uf1s":  fold_uf1s,
        "fold_uars":  fold_uars,
        "all_preds":  all_preds,
        "all_labels": all_labels,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = pd.read_csv(MANIFEST_PATH)
    df["emotion"] = df["emotion"].str.strip().str.lower()
    label_map  = fit_label_map(df["emotion"])
    class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]

    print(f"Device: {DEVICE}")
    print(f"Manifest clips: {len(df)}, Folds: {sorted(df['fold'].unique())}")
    print(f"Classes: {class_names}")
    print()

    # -----------------------------------------------------------------------
    # Evaluate all model families
    # -----------------------------------------------------------------------
    results = []

    # A. Original Three-Stream + Focal Loss champion
    sep = "=" * 72
    print(sep)
    print("A. ORIGINAL THREE-STREAM + FOCAL LOSS  (casme3_best_threestream_foldX.pt)")
    print(sep)
    r = evaluate_canonical(
        label          = "A: Three-Stream+Focal (ImageNet init)",
        ckpt_pattern   = "casme3_best_threestream_fold{fold}.pt",
        model_loader_fn= load_threestream_model,
        df=df, label_map=label_map, class_names=class_names,
    )
    results.append(r)

    # B. ThreeStreamModelV2 (the eval_output.txt champion, higher-capacity fusion)
    print()
    print(sep)
    print("B. THREE-STREAM V2  (casme3_best_threestream_v2_foldX.pt)")
    print(sep)
    r = evaluate_canonical(
        label          = "B: Three-Stream V2",
        ckpt_pattern   = "casme3_best_threestream_v2_fold{fold}.pt",
        model_loader_fn= load_threestream_v2_model,
        df=df, label_map=label_map, class_names=class_names,
    )
    results.append(r)

    # C. CASME2-pretrained RGB init Three-Stream V2 champion
    print()
    print(sep)
    print("C. CASME2-PRETRAINED THREE-STREAM V2 (casme3_best_threestream_v2_casme2_foldX.pt)")
    print(sep)
    r = evaluate_canonical(
        label          = "C: Three-Stream V2 (CASME2 RGB init)",
        ckpt_pattern   = "casme3_best_threestream_v2_casme2_fold{fold}.pt",
        model_loader_fn= load_threestream_v2_model,
        df=df, label_map=label_map, class_names=class_names,
    )
    results.append(r)

    # -----------------------------------------------------------------------
    # Final comparison table
    # -----------------------------------------------------------------------
    print()
    print(sep)
    print("CANONICAL COMPARISON TABLE (Single Source of Truth)")
    print(f"Canonical metric: Mean UF1 = avg of per-fold macro-F1 across 5 LOSO folds")
    print(sep)
    hdr = f"{'Model':<45} {'Mean UF1':>10} {'±':>4} {'Mean UAR':>10} {'±':>4} {'Global UF1':>12} {'Acc':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['label']:<45} "
              f"{r['mean_uf1']:>10.4f} "
              f"{r['std_uf1']:>4.4f} "
              f"{r['mean_uar']:>10.4f} "
              f"{r['std_uar']:>4.4f} "
              f"{r['global_uf1']:>12.4f} "
              f"{r['global_acc']:>8.4f}")
    print("-" * len(hdr))

    print()
    print(sep)
    print("PER-CLASS CLASSIFICATION REPORTS")
    print(sep)
    for r in results:
        print(f"\n--- {r['label']} ---")
        print(classification_report(
            r["all_labels"], r["all_preds"],
            target_names=class_names, zero_division=0
        ))

    # -----------------------------------------------------------------------
    # Save results to CSV
    # -----------------------------------------------------------------------
    rows = []
    for r in results:
        for i, (uf1, uar) in enumerate(zip(r["fold_uf1s"], r["fold_uars"])):
            rows.append({"model": r["label"], "fold": i, "fold_uf1": uf1, "fold_uar": uar})
    pd.DataFrame(rows).to_csv("canonical_eval_results.csv", index=False)
    print("\nPer-fold results saved to canonical_eval_results.csv")


if __name__ == "__main__":
    main()
