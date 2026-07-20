import pandas as pd
import os

manifest_path = r'D:\CASME PROJECT FINAL\CASME2_static_image_pool\casme2_manifest.csv'
df = pd.read_csv(manifest_path)
unique_subs = sorted(df['subject'].unique())
sub_to_fold = {sub: i for i, sub in enumerate(unique_subs)}

df['fold'] = df['subject'].map(sub_to_fold)
df.to_csv(manifest_path, index=False)

print(f"Total Folds: {df['fold'].nunique()}")
print("Subjects per fold mapping:")
for fold_idx in sorted(df['fold'].unique()):
    subs_in_fold = df[df['fold'] == fold_idx]['subject'].unique()
    print(f"Fold {fold_idx}: {subs_in_fold}")
