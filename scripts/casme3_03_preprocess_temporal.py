"""
03_preprocess_temporal.py -- For each clip, samples N evenly-spaced frames
between onset and offset (falling back to onset-apex range if offset frame
is missing), resizes, and caches as a single stacked array per clip.
This captures the full motion trajectory instead of just onset->apex flow.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

MANIFEST_PATH = Path("manifest.csv")
TEMPORAL_DIR = Path("processed_temporal")
IMG_SIZE = 128  # smaller than 224 since we now store many frames per clip (memory/speed)
NUM_FRAMES = 12  # evenly sampled across onset->offset window

TEMPORAL_DIR.mkdir(exist_ok=True)

def load_and_resize(path: Path, size: int):
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return img

def main():
    df = pd.read_csv(MANIFEST_PATH)
    print(f"Processing {len(df)} clips into temporal sequences...")

    n_ok, n_fail = 0, 0
    fail_clips = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        clip_name = row["clip_name"]
        out_path = TEMPORAL_DIR / f"{clip_name}.npz"
        if out_path.exists():
            n_ok += 1
            continue

        frame_folder = Path(row["frame_folder"])
        onset = int(row["onset"])
        offset = int(row["offset"])
        apex = int(row["apex"])

        start = onset
        end = offset if offset > onset else apex
        if end <= start:
            end = start + NUM_FRAMES  # fallback safety

        # Determine which frame indices actually exist in the folder
        available = sorted(int(p.stem) for p in frame_folder.glob("*.jpg"))
        if not available:
            n_fail += 1
            fail_clips.append(clip_name)
            continue

        # Sample NUM_FRAMES evenly across [start, end], clipped to available range
        target_indices = np.linspace(start, end, NUM_FRAMES).astype(int)
        # For each target, find the closest available frame
        available_arr = np.array(available)
        chosen = []
        for t in target_indices:
            closest = available_arr[np.argmin(np.abs(available_arr - t))]
            chosen.append(closest)

        frames = []
        ok = True
        for frame_num in chosen:
            frame_path = frame_folder / f"{frame_num}.jpg"
            img = load_and_resize(frame_path, IMG_SIZE)
            if img is None:
                ok = False
                break
            frames.append(img)

        if not ok or len(frames) != NUM_FRAMES:
            n_fail += 1
            fail_clips.append(clip_name)
            continue

        frames_arr = np.stack(frames, axis=0)  # (T, H, W, 3) uint8

        np.savez_compressed(out_path, frames=frames_arr, emotion=row["emotion"])
        n_ok += 1

    print(f"\nDone. OK: {n_ok}  Failed: {n_fail}")
    if fail_clips:
        print("Failed clips (first 20):", fail_clips[:20])

if __name__ == "__main__":
    main()
