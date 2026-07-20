import os
import zipfile
import pandas as pd
import collections
import numpy as np
import shutil
from sklearn.model_selection import StratifiedKFold

def main():
    print("Loading CASME2 metadata and preparing 5-fold CV...")
    
    excel_path = r"D:\CASME PROJECT FINAL\CASME 2 Dataset\CASME2-coding-20140508.xlsx"
    zip_path = r"D:\CASME PROJECT FINAL\CASME 2 Dataset\Cropped.zip"
    output_dir = r"D:\CASME PROJECT FINAL\CASME2_5fold_pool\images"
    manifest_path = r"D:\CASME PROJECT FINAL\manifest_casme2_5fold.csv"
    
    df = pd.read_excel(excel_path)
    
    # 1. 5 Classes ONLY: disgust, happy, others, repression, surprise
    emotion_map = {
        'happiness': 'happy',
        'disgust': 'disgust',
        'surprise': 'surprise',
        'others': 'others',
        'repression': 'repression'
    }
    # Note: 'sadness' and 'fear' are ignored entirely.
    
    clip_metadata = {}
    for idx, row in df.iterrows():
        sub = f"sub{int(row['Subject']):02d}"
        clip = str(row['Filename']).strip()
        raw_emotion = str(row['Estimated Emotion']).strip().lower()
        if raw_emotion in emotion_map:
            mapped_emotion = emotion_map[raw_emotion]
            key = f"{sub}/{clip}"
            clip_metadata[key] = mapped_emotion

    print("Reading Cropped.zip...")
    z = zipfile.ZipFile(zip_path)
    imgs = [f for f in z.namelist() if f.endswith('.jpg')]
    
    clip_to_images = collections.defaultdict(list)
    for img_path in imgs:
        parts = img_path.split('/')
        if len(parts) > 2:
            key = f"{parts[1]}/{parts[2]}"
            if key in clip_metadata:
                clip_to_images[key].append(img_path)
                
    # --- Prepare Stratified 5-Fold Split (by clip) ---
    clips = list(clip_to_images.keys())
    labels = [clip_metadata[c] for c in clips]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    clip_to_fold = {}
    for fold, (train_idx, val_idx) in enumerate(skf.split(clips, labels)):
        for idx in val_idx:
            clip_to_fold[clips[idx]] = fold

    os.makedirs(output_dir, exist_ok=True)
    
    manifest_rows = []
    
    print(f"Extracting frames to {output_dir} and generating manifest...")
    
    for clip, frames in clip_to_images.items():
        emotion = clip_metadata[clip]
        fold = clip_to_fold[clip]
        
        for img_path in frames:
            filename = img_path.replace('/', '_')
            dest = os.path.join(output_dir, filename)
            
            # Extract
            with z.open(img_path) as source, open(dest, 'wb') as target:
                shutil.copyfileobj(source, target)
                
            manifest_rows.append({
                'filename': filename,
                'clip_id': clip,
                'emotion': emotion,
                'fold': fold
            })
            
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(manifest_path, index=False)
    
    print(f"\nExtraction complete! Saved manifest to {manifest_path}")
    
    print("\n=== FINAL CLASS DISTRIBUTION ===")
    print(manifest_df['emotion'].value_counts())
    
    print("\n=== FOLD DISTRIBUTION (Clips) ===")
    clip_df = manifest_df.drop_duplicates(subset=['clip_id'])
    for f in range(5):
        print(f"Fold {f}:")
        print(clip_df[clip_df['fold'] == f]['emotion'].value_counts())

if __name__ == "__main__":
    main()
