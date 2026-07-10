"""
04b_sanity_check_rois.py -- Visualize the 4 cropped ROI patches for several
random clips. This is the check we should have done BEFORE training the
HTNet-lite model. Look for: do eye patches actually show eyes? Do mouth
patches show mouth halves that are different (not duplicates)? Any patches
mostly black/empty (failed crop)?
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MANIFEST_PATH = Path("manifest.csv")
ROI_DIR = Path("processed_roi")
OUT_PATH = Path("roi_sanity_check.png")
N_SAMPLES = 6
REGION_NAMES = ["left_eye", "right_eye", "mouth"]

def flow_to_gray(flow):
    mag = np.hypot(flow[..., 0], flow[..., 1])
    return mag

def main():
    df = pd.read_csv(MANIFEST_PATH)
    df["roi_exists"] = df["clip_name"].apply(lambda c: (ROI_DIR / f"{c}.npz").exists())
    df = df[df["roi_exists"]].reset_index(drop=True)
    sample = df.sample(n=N_SAMPLES, random_state=7)

    fig, axes = plt.subplots(N_SAMPLES, 6, figsize=(12, 2.2 * N_SAMPLES))

    for i, (_, row) in enumerate(sample.iterrows()):
        data = np.load(ROI_DIR / f"{row['clip_name']}.npz")
        rgb_patches = data["rgb_patches"]    # (3, 24, 40, 3)
        flow_patches = data["flow_patches"]  # (3, 24, 40, 2)

        for r in range(3):
            axes[i, r].imshow(rgb_patches[r])
            axes[i, r].set_title(f"{REGION_NAMES[r]}\nrgb", fontsize=7)
            axes[i, r].axis("off")

            axes[i, r + 3].imshow(flow_to_gray(flow_patches[r]), cmap="hot")
            mean_mag = flow_to_gray(flow_patches[r]).mean()
            axes[i, r + 3].set_title(f"flow\nmag={mean_mag:.2f}", fontsize=7)
            axes[i, r + 3].axis("off")

        axes[i, 0].set_ylabel(f"{row['clip_name']}\n{row['emotion']}", fontsize=7)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=110)
    print(f"Saved to {OUT_PATH.resolve()}")

    print("\nChecking for degenerate (all-black or identical) patches across 50 random clips...")
    check_sample = df.sample(n=min(50, len(df)), random_state=1)
    n_zero_patches = 0
    for _, row in check_sample.iterrows():
        data = np.load(ROI_DIR / f"{row['clip_name']}.npz")
        rgb_patches = data["rgb_patches"]
        for r in range(3):
            if rgb_patches[r].sum() == 0:
                n_zero_patches += 1
    print(f"All-black patches found: {n_zero_patches} (out of {len(check_sample)*3} patches checked)")

    print("\nFlow magnitude stats per region across 100 random clips:")
    stat_sample = df.sample(n=min(100, len(df)), random_state=2)
    region_mags = {name: [] for name in REGION_NAMES}
    for _, row in stat_sample.iterrows():
        data = np.load(ROI_DIR / f"{row['clip_name']}.npz")
        flow_patches = data["flow_patches"]
        for r in range(3):
            region_mags[REGION_NAMES[r]].append(flow_to_gray(flow_patches[r]).mean())
    for name, mags in region_mags.items():
        print(f"  {name}: mean={np.mean(mags):.4f}  std={np.std(mags):.4f}  max={np.max(mags):.4f}")

if __name__ == "__main__":
    main()