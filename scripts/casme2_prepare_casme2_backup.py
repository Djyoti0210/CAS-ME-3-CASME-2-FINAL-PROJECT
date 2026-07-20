import os
import zipfile
import pandas as pd
import collections
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
# No torch imports here to avoid environment issues

def main():
    print("Loading CASME2 static image pool...")
    
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
            clip_metadata[key] = mapped_emotion

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
    for clip, frames in clip_to_images.items():
        emotion = clip_metadata[clip]
        class_frame_counts[emotion].append(len(frames))
        
    print("\n--- Clip Length Check ---")
    for emotion, counts in class_frame_counts.items():
        avg = np.mean(counts)
        total = np.sum(counts)
        print(f"Class '{emotion}': {len(counts)} clips, {total} images (Avg {avg:.1f} images/clip)")
        if avg > 100:
            print(f"  >>> LOG: Class '{emotion}' has a suspiciously high number of images per clip (avg {avg:.1f}).")
    print("-------------------------\n")

    # --- Prepare Stratified Split (by clip) ---
    # We split by clip to avoid frame-level leakage, which is more robust even without subject-independence
    clips = list(clip_to_images.keys())
    labels = [clip_metadata[c] for c in clips]
    
    train_clips, holdout_clips = train_test_split(clips, test_size=0.15, stratify=labels, random_state=42)
    
    train_clips_set = set(train_clips)
    holdout_clips_set = set(holdout_clips)

    # Calculate final distribution
    train_counts = collections.defaultdict(int)
    holdout_counts = collections.defaultdict(int)
    
    total_train = 0
    total_holdout = 0
    
    print(f"Extracting frames to {output_dir}...")
    
    for split in ['train', 'holdout']:
        for emotion in set(emotion_map.values()):
            os.makedirs(os.path.join(output_dir, split, emotion), exist_ok=True)
            
    # Extract
    for clip, frames in clip_to_images.items():
        emotion = clip_metadata[clip]
        if clip in train_clips_set:
            split = 'train'
            train_counts[emotion] += len(frames)
            total_train += len(frames)
        else:
            split = 'holdout'
            holdout_counts[emotion] += len(frames)
            total_holdout += len(frames)
            
        for img_path in frames:
            # We extract it directly and flat to the emotion folder, with a unique name
            filename = img_path.replace('/', '_')
            dest = os.path.join(output_dir, split, emotion, filename)
            
            with z.open(img_path) as source, open(dest, 'wb') as target:
                shutil.copyfileobj(source, target)
                
    print("\nExtraction complete! NOTE: This pretraining stage does NOT have subject-independence guarantees.")
    
    print("\n=== FINAL CLASS DISTRIBUTION ===")
    print(f"{'Class':<10} | {'Train':<7} | {'Holdout':<7} | {'Total':<7}")
    print("-" * 35)
    for emotion in sorted(set(emotion_map.values())):
        tr = train_counts[emotion]
        ho = holdout_counts[emotion]
        print(f"{emotion:<10} | {tr:<7} | {ho:<7} | {tr+ho:<7}")
    print("-" * 35)
    print(f"{'TOTAL':<10} | {total_train:<7} | {total_holdout:<7} | {total_train+total_holdout:<7}")
    
    # Show Dataset/DataLoader structure code
    print("\n=== ImageFolder & DataLoader Setup Example ===")
    print("transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])")
    print(f"train_dataset = datasets.ImageFolder('{output_dir}/train', transform=transform)")
    print(f"holdout_dataset = datasets.ImageFolder('{output_dir}/holdout', transform=transform)")
    print("train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)")
    print("holdout_loader = DataLoader(holdout_dataset, batch_size=32, shuffle=False)")

if __name__ == "__main__":
    main()
