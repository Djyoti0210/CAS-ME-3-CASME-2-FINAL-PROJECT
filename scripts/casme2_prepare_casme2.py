import os
import zipfile
import pandas as pd
import collections
import numpy as np
import shutil
# No torch imports here to avoid environment issues

def main():
    print("Loading CASME2 static image pool (LOSO-compliant)...")
    
    excel_path = r"D:\CASME PROJECT FINAL\CASME 2 Dataset\CASME2-coding-20140508.xlsx"
    zip_path = r"D:\CASME PROJECT FINAL\CASME 2 Dataset\Cropped.zip"
    output_dir = r"D:\CASME PROJECT FINAL\CASME2_static_image_pool"
    
    df = pd.read_excel(excel_path)
    
    # Remap classes
    emotion_map = {
        'happiness': 'happy',
        'sadness': 'sad',
        'disgust': 'disgust',
        'surprise': 'surprise',
        'fear': 'fear',
        'others': 'others',
        'repression': 'others'
    }
    
    # Process metadata to get valid clips and their mapped emotions
    clip_metadata = {}
    for idx, row in df.iterrows():
        sub = f"sub{int(row['Subject']):02d}"
        clip = str(row['Filename']).strip()
        raw_emotion = str(row['Estimated Emotion']).strip().lower()
        if raw_emotion in emotion_map:
            mapped_emotion = emotion_map[raw_emotion]
            key = f"{sub}/{clip}"
            # Store tuple: (subject, clip, mapped_emotion)
            clip_metadata[key] = (sub, clip, mapped_emotion)

    # Read zip file to count and map images
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
                
    # --- Check for suspicious images per clip (average per class) ---
    class_frame_counts = collections.defaultdict(list)
    for key, frames in clip_to_images.items():
        _, _, emotion = clip_metadata[key]
        class_frame_counts[emotion].append(len(frames))
        
    print("\n--- Clip Length Check ---")
    for emotion, counts in class_frame_counts.items():
        avg = np.mean(counts)
        total = np.sum(counts)
        print(f"Class '{emotion}': {len(counts)} clips, {total} images (Avg {avg:.1f} images/clip)")
        if avg > 100:
            print(f"  >>> LOG: Class '{emotion}' has a suspiciously high number of images per clip (avg {avg:.1f}).")
    print("-------------------------\n")

    # --- Extract Images and Generate Manifest ---
    print(f"Extracting frames to {output_dir}...")
    
    images_output_dir = os.path.join(output_dir, "images")
    
    manifest_rows = []
    
    # Generate subject-to-fold mapping (True LOSO: 1 subject = 1 fold)
    unique_subs = sorted(list(set([sub for sub, _, _ in clip_metadata.values()])))
    sub_to_fold = {sub: i for i, sub in enumerate(unique_subs)}
    print(f"\nGenerated {len(unique_subs)} folds for True LOSO.")
    
    for key, frames in clip_to_images.items():
        sub, clip, emotion = clip_metadata[key]
        fold = sub_to_fold[sub]
        
        # Create output directory: CASME2_static_image_pool/images/<subject>/<emotion>/
        dest_dir = os.path.join(images_output_dir, sub, emotion)
        os.makedirs(dest_dir, exist_ok=True)
            
        for img_path in frames:
            # We extract it flat to the emotion folder, with a unique name
            filename = img_path.replace('/', '_')
            dest = os.path.join(dest_dir, filename)
            
            # Relative path for manifest
            rel_path = f"{sub}/{emotion}/{filename}"
            
            manifest_rows.append({
                "image_path": rel_path,
                "subject": sub,
                "clip": clip,
                "emotion": emotion,
                "fold": fold
            })
            
            with z.open(img_path) as source, open(dest, 'wb') as target:
                shutil.copyfileobj(source, target)
                
    print("\nExtraction complete! Data is structured for LOSO cross-validation.")
    
    # Save manifest
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(output_dir, "casme2_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Manifest saved to {manifest_path}")
    print(f"Total extracted images: {len(manifest_df)}")
    
    # Display final distribution
    print("\n=== FINAL CLASS DISTRIBUTION ===")
    dist = manifest_df['emotion'].value_counts()
    print(dist)
    print("-------------------------")
    print(f"TOTAL: {dist.sum()}")
    
    # Show Dataset/DataLoader structure code for custom manifest-based loading
    print("\n=== Dataset loading can now use the casme2_manifest.csv ===")
    print("With columns: ['image_path', 'subject', 'clip', 'emotion', 'fold']")

if __name__ == "__main__":
    main()
