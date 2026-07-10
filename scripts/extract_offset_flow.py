import os
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
import time

dataset_path = r'D:\CASME PROJECT FINAL\CASME3_Dataset'
manifest_path = os.path.join(dataset_path, 'fold_manifest.csv')
out_dir = os.path.join(dataset_path, 'processed_offset')
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(manifest_path)
df = df[~df['clip_id'].isin(['spNO.39_j_855', 'spNO.40_j_1016'])] # Known missing face clips

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

ANCHOR_IDX = [
    133, 362, 168, 8, 9, 6, 
    33, 155, 157, 158, 159, 160, 161, 246, 
    263, 382, 384, 385, 386, 387, 388, 466, 
    124, 156, 113, 225, 224, 223, 222, 221, 189, 
    353, 383, 342, 445, 444, 443, 442, 441, 413
]

def get_face_bounding_box(landmarks, img_w, img_h, padding_ratio=0.1):
    xs = [lm.x * img_w for lm in landmarks]
    ys = [lm.y * img_h for lm in landmarks]
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

print(f"Starting Onset->Offset Flow Extraction for {len(df)} clips...")
start_time = time.time()
processed = 0

for idx, row in df.iterrows():
    cid = row['clip_id']
    f_dir = row['frame_dir']
    
    if idx % 100 == 0:
        print(f"Processing clip {idx}/{len(df)}...")
        
    on = int(row['Onset'])
    off = int(row['Offset'])
    if on == off: off = on + 1
        
    on_path = find_frame(f_dir, on)
    off_path = find_frame(f_dir, off)
    if not on_path or not off_path: continue
        
    img_on = cv2.imread(on_path)
    img_off = cv2.imread(off_path)
    if img_on is None or img_off is None: continue
    h, w, _ = img_on.shape
    
    res_on = face_mesh.process(cv2.cvtColor(img_on, cv2.COLOR_BGR2RGB))
    res_off = face_mesh.process(cv2.cvtColor(img_off, cv2.COLOR_BGR2RGB))
    if not res_on.multi_face_landmarks or not res_off.multi_face_landmarks: continue
        
    lms_on = res_on.multi_face_landmarks[0].landmark
    lms_off = res_off.multi_face_landmarks[0].landmark
    
    pts_on = np.float32([[lms_on[i].x * w, lms_on[i].y * h] for i in ANCHOR_IDX])
    pts_off = np.float32([[lms_off[i].x * w, lms_off[i].y * h] for i in ANCHOR_IDX])
    
    M, inliers = cv2.estimateAffinePartial2D(pts_off, pts_on, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    
    if M is None:
        M = np.float32([[1, 0, 0], [0, 1, 0]])
            
    img_off_aligned = cv2.warpAffine(img_off, M, (w, h))
    
    xmin, ymin, xmax, ymax = get_face_bounding_box(lms_on, w, h)
    
    gray_on = cv2.cvtColor(img_on, cv2.COLOR_BGR2GRAY)
    gray_off_aligned = cv2.cvtColor(img_off_aligned, cv2.COLOR_BGR2GRAY)
    
    crop_on = gray_on[ymin:ymax, xmin:xmax]
    crop_off_aligned = gray_off_aligned[ymin:ymax, xmin:xmax]
    
    crop_on_rs = cv2.resize(crop_on, (224, 224), interpolation=cv2.INTER_LINEAR)
    crop_off_rs = cv2.resize(crop_off_aligned, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    flow_offset = cv2.calcOpticalFlowFarneback(crop_on_rs, crop_off_rs, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    np.savez_compressed(
        os.path.join(out_dir, f"{cid}.npz"), 
        flow_offset=flow_offset
    )
    processed += 1

print(f"Extraction complete. {processed} clips processed in {time.time()-start_time:.1f} seconds")
