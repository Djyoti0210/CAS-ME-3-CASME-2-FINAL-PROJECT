import pandas as pd
import os

excel_path = r"D:\CASME PROJECT FINAL\CASME 2 Dataset\CASME2-coding-20140508.xlsx"
manifest_path = r"D:\CASME PROJECT FINAL\CASME2_static_image_pool\casme2_manifest.csv"

# 1. Load Excel and build true mapping
df_excel = pd.read_excel(excel_path)
true_mapping = {}
for idx, row in df_excel.iterrows():
    sub = f"sub{int(row['Subject']):02d}"
    clip = str(row['Filename']).strip()
    raw_emotion = str(row['Estimated Emotion']).strip().lower()
    
    if raw_emotion == 'happiness':
        mapped_emotion = 'happy'
    elif raw_emotion in ['disgust', 'others', 'repression', 'surprise']:
        mapped_emotion = raw_emotion
    else:
        mapped_emotion = 'DROP'
        
    true_mapping[f"{sub}/{clip}"] = mapped_emotion

# 2. Patch manifest
df_manifest = pd.read_csv(manifest_path)
print(f"Original manifest rows: {len(df_manifest)}")

new_emotions = []
for idx, row in df_manifest.iterrows():
    key = f"{row['subject']}/{row['clip']}"
    if key in true_mapping:
        new_emotions.append(true_mapping[key])
    else:
        new_emotions.append('DROP')
        
df_manifest['emotion'] = new_emotions

# 3. Filter out DROP rows
df_manifest = df_manifest[df_manifest['emotion'] != 'DROP'].copy()
print(f"Patched manifest rows (5-class): {len(df_manifest)}")

df_manifest.to_csv(manifest_path, index=False)
print("Manifest updated successfully.")
print(df_manifest['emotion'].value_counts())
