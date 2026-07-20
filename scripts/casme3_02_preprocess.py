"""
02_preprocess.py -- For every clip in manifest.csv, loads onset+apex jpg frames,
computes dense optical flow between them, resizes RGB(apex)+flow, and caches
to a .npz file per clip. Run once; training reads only from these .npz files.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

MANIFEST_PATH = Path("manifest.csv")
PROCESSED_DIR = Path("processed")
IMG_SIZE = 224  # resize target (matches ResNet18 input expectations)

PROCESSED_DIR.mkdir(exist_ok=True)

def load_and_resize(path: Path, size: int):
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.resize(img, (size, size))
    return img

def compute_flow(onset_gray, apex_gray):
    flow = cv2.calcOpticalFlowFarneback(
        onset_gray, apex_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow  # (H, W, 2) -- x and y flow components

def main():
    df = pd.read_csv(MANIFEST_PATH)
    print(f"Processing {len(df)} clips...")

    n_ok, n_fail = 0, 0
    fail_clips = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        clip_name = row["clip_name"]
        out_path = PROCESSED_DIR / f"{clip_name}.npz"
        if out_path.exists():
            n_ok += 1
            continue  # already processed, skip (safe to re-run this script)

        frame_folder = Path(row["frame_folder"])
        onset_path = frame_folder / f"{row['onset']}.jpg"
        apex_path = frame_folder / f"{row['apex']}.jpg"

        onset_img = load_and_resize(onset_path, IMG_SIZE)
        apex_img = load_and_resize(apex_path, IMG_SIZE)

        if onset_img is None or apex_img is None:
            n_fail += 1
            fail_clips.append(clip_name)
            continue

        onset_gray = cv2.cvtColor(onset_img, cv2.COLOR_BGR2GRAY)
        apex_gray = cv2.cvtColor(apex_img, cv2.COLOR_BGR2GRAY)
        flow = compute_flow(onset_gray, apex_gray)  # (H, W, 2)

        # apex_img is BGR uint8 (H, W, 3); flow is float32 (H, W, 2)
        apex_rgb = cv2.cvtColor(apex_img, cv2.COLOR_BGR2RGB)

        np.savez_compressed(
            out_path,
            rgb=apex_rgb.astype(np.uint8),
            flow=flow.astype(np.float32),
            emotion=row["emotion"],
        )
        n_ok += 1

    print(f"\nDone. OK: {n_ok}  Failed: {n_fail}")
    if fail_clips:
        print("Failed clips (first 20):", fail_clips[:20])

if __name__ == "__main__":
    main()
