import zipfile
import pandas as pd
import collections

z = zipfile.ZipFile('d:/CASME PROJECT FINAL/CASME 2 Dataset/Cropped.zip')
imgs = [f for f in z.namelist() if f.endswith('.jpg')]
counts = collections.Counter([f.split('/')[1] + '/' + f.split('/')[2] for f in imgs if len(f.split('/'))>2])

df = pd.read_excel('d:/CASME PROJECT FINAL/CASME 2 Dataset/CASME2-coding-20140508.xlsx')
total_imgs = 0
for idx, row in df.iterrows():
    sub = f"sub{int(row['Subject']):02d}"
    clip = str(row['Filename']).strip()
    emotion = str(row['Estimated Emotion']).strip().lower()
    key = f"{sub}/{clip}"
    count = counts.get(key, 0)
    total_imgs += count

print(f"Total labeled images: {total_imgs}")
