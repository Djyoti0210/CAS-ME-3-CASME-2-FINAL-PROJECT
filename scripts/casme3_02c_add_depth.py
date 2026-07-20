"""
02c_add_depth.py -- Adds the apex-frame depth image into the existing cached
.npz files (alongside rgb, flow, emotion). Does NOT recompute rgb/flow --
just reads depth/ and re-saves each npz with the extra array.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

MANIFEST_PATH = Path("manifest.csv")
PROCESSED_DIR = Path("processed")
IMG_SIZE = 224

def main():
    df = pd.read_csv(MANIFEST_PATH)
    print(f"Adding depth to {len(df)} clips...")

    n_ok, n_fail, n_skip = 0, 0, 0
    fail_clips = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        clip_name = row["clip_name"]
        npz_path = PROCESSED_DIR / f"{clip_name}.npz"

        if not npz_path.exists():
            n_fail += 1
            fail_clips.append(clip_name)
            continue

        existing = dict(np.load(npz_path))
        if "depth" in existing:
            n_skip += 1
            continue  # already has depth, skip (resumable)

        depth_folder = Path(row["depth_folder"])
        apex_depth_path = depth_folder / f"{row['apex']}.png"

        depth_img = cv2.imread(str(apex_depth_path), cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            n_fail += 1
            fail_clips.append(clip_name)
            continue

        depth_img = cv2.resize(depth_img, (IMG_SIZE, IMG_SIZE))
        if depth_img.ndim == 3:
            depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)

        existing["depth"] = depth_img.astype(np.uint8)
        np.savez_compressed(npz_path, **existing)
        n_ok += 1

    print(f"\nDone. Added: {n_ok}  Skipped(already had): {n_skip}  Failed: {n_fail}")
    if fail_clips:
        print("Failed clips (first 20):", fail_clips[:20])

if __name__ == "__main__":
    main()
