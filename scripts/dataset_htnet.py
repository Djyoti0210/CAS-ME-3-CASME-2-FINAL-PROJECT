"""
dataset_htnet.py -- Loads the 4 cropped ROI patches (RGB+Flow) per clip
for the lightweight HTNet-style model.
"""
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

ROI_DIR = Path("processed_roi")

def fit_label_map(emotions):
    unique = sorted(emotions.unique())
    return {e: i for i, e in enumerate(unique)}

class ROIDataset(Dataset):
    def __init__(self, df, label_map, is_train=False):
        self.df = df.reset_index(drop=True)
        self.label_map = label_map
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npz_path = ROI_DIR / f"{row['clip_name']}.npz"
        data = np.load(npz_path)

        rgb_patches = data["rgb_patches"].copy().astype(np.float32) / 255.0   # (4, 28, 28, 3)
        flow_patches = data["flow_patches"].copy()                            # (4, 28, 28, 2)
        flow_patches = np.clip(flow_patches / 15.0, -1.0, 1.0)

        if self.is_train and random.random() > 0.5:
            # Horizontal flip: swap left_eye <-> right_eye, mouth flips in place
            rgb_patches = np.flip(rgb_patches, axis=2).copy()
            flow_patches = np.flip(flow_patches, axis=2).copy()
            flow_patches[..., 0] = -flow_patches[..., 0]
            rgb_patches = rgb_patches[[1, 0, 2]]   # swap eyes, mouth (index 2) stays
            flow_patches = flow_patches[[1, 0, 2]]

        # Concatenate RGB + Flow channels per patch -> 5 channels
        combined = np.concatenate([rgb_patches, flow_patches], axis=-1)  # (4, 28, 28, 5)
        combined_t = torch.from_numpy(combined).float().permute(0, 3, 1, 2)  # (4, 5, 28, 28)

        label = self.label_map[str(row["emotion"]).strip().lower()]
        return combined_t, torch.tensor(label, dtype=torch.long)

def build_dataloader(df, label_map, batch_size=16, shuffle=True, is_train=False, use_weighted_sampler=True):
    dataset = ROIDataset(df, label_map, is_train=is_train)

    sampler = None
    if shuffle and is_train and use_weighted_sampler:
        class_counts = df["emotion"].value_counts()
        total = len(df)
        weights_dict = {cls: total / count for cls, count in class_counts.items()}
        sample_weights = [weights_dict[str(e).strip().lower()] for e in df["emotion"]]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=0, pin_memory=True
    )