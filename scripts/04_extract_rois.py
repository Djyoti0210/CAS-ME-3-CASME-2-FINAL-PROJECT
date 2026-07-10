"""
04_extract_rois.py -- HTNet-style preprocessing. For each clip, runs MediaPipe
FaceMesh on the apex RGB frame to locate eye/lip regions, then crops those same
regions out of both the RGB and Flow arrays (already cached at 224x224).
Saves 4 small patches (left_eye, right_eye, mouth_left, mouth_right) per clip.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import mediapipe as mp

MANIFEST_PATH = Path("manifest.csv")
PROCESSED_DIR = Path("processed")       # existing rgb/flow/depth npz
ROI_DIR = Path("processed_roi")         # new output
ROI_DIR.mkdir(exist_ok=True)

PATCH_SIZE = 28  # each cropped region resized to 28x28 (small, per HTNet-style design)
IMG_SIZE = 224   # matches existing cached rgb/flow size

LEFT_EYE_ALL = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_ALL = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
MOUTH_FULL = [61, 291, 39, 181, 0, 17, 269, 405, 314, 84, 178, 88, 78, 308, 13, 14]

mp_face_mesh = mp.solutions.face_mesh
PATCH_H, PATCH_W = 24, 40  # rectangular, matches natural eye/mouth aspect ratio -- no wasted padding

def get_bbox(landmarks, idx_list, img_w, img_h, pad_ratio=0.4):
    pts = np.array([(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in idx_list])
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    w, h = x_max - x_min, y_max - y_min
    pad_w, pad_h = w * pad_ratio, h * pad_ratio
    x_min = max(0, int(x_min - pad_w))
    y_min = max(0, int(y_min - pad_h))
    x_max = min(img_w, int(x_max + pad_w))
    y_max = min(img_h, int(y_max + pad_h))
    return x_min, y_min, x_max, y_max

def crop_and_resize_rect(img, bbox, out_h, out_w):
    """Direct resize to a fixed rectangular target -- no letterbox padding,
    uses full resolution, accepts mild aspect distortion (much less severe
    than the black-padding waste from square letterboxing)."""
    x_min, y_min, x_max, y_max = bbox
    if x_max <= x_min or y_max <= y_min:
        return np.zeros((out_h, out_w, img.shape[2]), dtype=img.dtype)
    crop = img[y_min:y_max, x_min:x_max]
    return cv2.resize(crop, (out_w, out_h))

def main():
    df = pd.read_csv(MANIFEST_PATH)
    print(f"Extracting ROIs for {len(df)} clips...")

    n_ok, n_fail = 0, 0
    fail_clips = []

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                 refine_landmarks=False, min_detection_confidence=0.3) as face_mesh:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            clip_name = row["clip_name"]
            out_path = ROI_DIR / f"{clip_name}.npz"
            if out_path.exists():
                n_ok += 1
                continue

            npz_path = PROCESSED_DIR / f"{clip_name}.npz"
            if not npz_path.exists():
                n_fail += 1
                fail_clips.append(clip_name)
                continue

            data = np.load(npz_path)
            rgb = data["rgb"]    # (224, 224, 3) uint8
            flow = data["flow"]  # (224, 224, 2) float32

            results = face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                n_fail += 1
                fail_clips.append(clip_name)
                continue

            landmarks = results.multi_face_landmarks[0].landmark

            regions = {
                "left_eye": LEFT_EYE_ALL,
                "right_eye": RIGHT_EYE_ALL,
                "mouth": MOUTH_FULL,
            }

            patches_rgb = []
            patches_flow = []
            for region_name, idx_list in regions.items():
                bbox = get_bbox(landmarks, idx_list, IMG_SIZE, IMG_SIZE)
                rgb_patch = crop_and_resize_rect(rgb, bbox, PATCH_H, PATCH_W)
                flow_patch = crop_and_resize_rect(flow, bbox, PATCH_H, PATCH_W)
                patches_rgb.append(rgb_patch)
                patches_flow.append(flow_patch)

            patches_rgb = np.stack(patches_rgb, axis=0)    # (4, 28, 28, 3)
            patches_flow = np.stack(patches_flow, axis=0)  # (4, 28, 28, 2)

            np.savez_compressed(out_path, rgb_patches=patches_rgb, flow_patches=patches_flow,
                                 emotion=row["emotion"])
            n_ok += 1

    print(f"\nDone. OK: {n_ok}  Failed: {n_fail}")
    if fail_clips:
        print("Failed clips (first 20):", fail_clips[:20])
        print(f"Failure rate: {n_fail}/{len(df)} = {100*n_fail/len(df):.1f}%")

if __name__ == "__main__":
    main()