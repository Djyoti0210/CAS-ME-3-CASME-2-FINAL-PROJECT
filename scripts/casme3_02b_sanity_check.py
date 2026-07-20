"""
02b_sanity_check.py -- Loads a few random .npz files and saves a visualization
so you can eyeball that RGB looks like a real face and flow shows motion
concentrated around eyes/brows/mouth (not noise everywhere or all-zero).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MANIFEST_PATH = Path("manifest.csv")
PROCESSED_DIR = Path("processed")
OUT_PATH = Path("sanity_check.png")
N_SAMPLES = 6

def flow_to_rgb(flow):
    """Convert (H,W,2) flow to a viewable RGB image using HSV encoding."""
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = np.hypot(flow[..., 0], flow[..., 1]), np.arctan2(flow[..., 1], flow[..., 0])
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 2] = np.clip(mag * 8, 0, 255).astype(np.uint8)
    import cv2
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def main():
    df = pd.read_csv(MANIFEST_PATH)
    sample = df.sample(n=N_SAMPLES, random_state=1)

    fig, axes = plt.subplots(2, N_SAMPLES, figsize=(3 * N_SAMPLES, 6))

    for i, (_, row) in enumerate(sample.iterrows()):
        npz_path = PROCESSED_DIR / f"{row['clip_name']}.npz"
        data = np.load(npz_path)
        rgb = data["rgb"]
        flow = data["flow"]

        axes[0, i].imshow(rgb)
        axes[0, i].set_title(f"{row['clip_name']}\n{row['emotion']}", fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(flow_to_rgb(flow))
        axes[1, i].set_title("flow", fontsize=8)
        axes[1, i].axis("off")

        print(f"{row['clip_name']}: flow magnitude mean={np.hypot(flow[...,0], flow[...,1]).mean():.4f}, "
              f"max={np.hypot(flow[...,0], flow[...,1]).max():.4f}")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=120)
    print(f"\nSaved visualization to {OUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
