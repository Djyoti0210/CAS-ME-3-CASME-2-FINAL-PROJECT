import os
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
import math
import time

dataset_path = r'D:\CASME PROJECT FINAL\CASME3_Dataset'
manifest_path = os.path.join(dataset_path, 'fold_manifest.csv')
out_dir = os.path.join(dataset_path, 'extracted_rois_v3')

df = pd.read_csv(manifest_path)
df = df[~df['clip_id'].isin(['spNO.39_j_855', 'spNO.40_j_1016'])] # Known missing face clips

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Anchor landmarks for RANSAC
ANCHOR_IDX = [
    133, 362, 168, 8, 9, 6, 
    33, 155, 157, 158, 159, 160, 161, 246, 
    263, 382, 384, 385, 386, 387, 388, 466, 
    124, 156, 113, 225, 224, 223, 222, 221, 189, 
    353, 383, 342, 445, 444, 443, 442, 441, 413
]

IDX_LEFT_EYEBROW = list(set([item for sublist in mp_face_mesh.FACEMESH_LEFT_EYEBROW for item in sublist]))
IDX_RIGHT_EYEBROW = list(set([item for sublist in mp_face_mesh.FACEMESH_RIGHT_EYEBROW for item in sublist]))
IDX_EYEBROWS = list(set(IDX_LEFT_EYEBROW + IDX_RIGHT_EYEBROW))

def get_bounding_box(landmarks, indices, img_w, img_h, padding_ratio=0.3):
    xs = [landmarks[i].x * img_w for i in indices]
    ys = [landmarks[i].y * img_h for i in indices]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    w, h = xmax - xmin, ymax - ymin
    pad_w, pad_h = w * padding_ratio, h * padding_ratio
    xmin = max(0, int(xmin - pad_w))
    xmax = min(img_w, int(xmax + pad_w))
    ymin = max(0, int(ymin - pad_h))
    ymax = min(img_h, int(ymax + pad_h))
    return xmin, ymin, xmax, ymax

def find_frame(f_dir, num):
    all_files = os.listdir(f_dir)
    for f in all_files:
        if f.split('.')[0].lstrip('0') == str(num).lstrip('0'): return os.path.join(f_dir, f)
        if f.split('.')[0] == str(num): return os.path.join(f_dir, f)
    return None

print(f"Starting Eyebrow Extraction for {len(df)} clips...")
start_time = time.time()
processed = 0

for idx, row in df.iterrows():
    cid = row['clip_id']
    f_dir = row['frame_dir']
    
    if idx % 100 == 0:
        print(f"Processing clip {idx}/{len(df)}...")
        
    on = int(row['Onset'])
    ap = int(row['Apex'])
    if on == ap: ap = on + 1
        
    on_path = find_frame(f_dir, on)
    ap_path = find_frame(f_dir, ap)
    if not on_path or not ap_path: continue
        
    img_on = cv2.imread(on_path)
    img_ap = cv2.imread(ap_path)
    if img_on is None or img_ap is None: continue
    h, w, _ = img_on.shape
    
    res_on = face_mesh.process(cv2.cvtColor(img_on, cv2.COLOR_BGR2RGB))
    res_ap = face_mesh.process(cv2.cvtColor(img_ap, cv2.COLOR_BGR2RGB))
    if not res_on.multi_face_landmarks or not res_ap.multi_face_landmarks: continue
        
    lms_on = res_on.multi_face_landmarks[0].landmark
    lms_ap = res_ap.multi_face_landmarks[0].landmark
    
    pts_on = np.float32([[lms_on[i].x * w, lms_on[i].y * h] for i in ANCHOR_IDX])
    pts_ap = np.float32([[lms_ap[i].x * w, lms_ap[i].y * h] for i in ANCHOR_IDX])
    
    M, inliers = cv2.estimateAffinePartial2D(pts_ap, pts_on, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    
    if M is None:
        M = np.float32([[1, 0, 0], [0, 1, 0]])
            
    img_ap_aligned = cv2.warpAffine(img_ap, M, (w, h))
    
    xmin, ymin, xmax, ymax = get_bounding_box(lms_on, IDX_EYEBROWS, w, h)
    
    gray_on = cv2.cvtColor(img_on, cv2.COLOR_BGR2GRAY)
    gray_ap_aligned = cv2.cvtColor(img_ap_aligned, cv2.COLOR_BGR2GRAY)
    
    crop_on = gray_on[ymin:ymax, xmin:xmax]
    crop_ap_aligned = gray_ap_aligned[ymin:ymax, xmin:xmax]
    crop_on_rgb = img_on[ymin:ymax, xmin:xmax]
    
    flow = cv2.calcOpticalFlowFarneback(crop_on, crop_ap_aligned, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    clip_out = os.path.join(out_dir, cid)
    os.makedirs(clip_out, exist_ok=True)
    
    np.savez_compressed(
        os.path.join(clip_out, "eyebrows.npz"), 
        rgb=cv2.cvtColor(crop_on_rgb, cv2.COLOR_BGR2RGB),
        flow=flow
    )
    processed += 1

print(f"Extraction complete. {processed} clips processed in {time.time()-start_time:.1f} seconds")
