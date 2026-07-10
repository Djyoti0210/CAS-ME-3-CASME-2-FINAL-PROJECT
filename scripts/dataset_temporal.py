"""
dataset_temporal.py -- Loads cached frame sequences (T, H, W, 3) for the
CNN+GRU temporal model. Separate from dataset.py (which handles the
single-frame RGB/Flow/Depth streams) so both can coexist.
"""
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

TEMPORAL_DIR = Path("processed_temporal")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def fit_label_map(emotions):
    unique = sorted(emotions.unique())
    return {e: i for i, e in enumerate(unique)}

class TemporalDataset(Dataset):
    def __init__(self, df, label_map, is_train=False):
        self.df = df.reset_index(drop=True)
        self.label_map = label_map
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npz_path = TEMPORAL_DIR / f"{row['clip_name']}.npz"
        data = np.load(npz_path)
        frames = data["frames"].copy().astype(np.float32) / 255.0  # (T, H, W, 3)

        if self.is_train and random.random() > 0.5:
            frames = np.flip(frames, axis=2).copy()  # horizontal flip on W axis

        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD
        frames_t = torch.from_numpy(frames).float().permute(0, 3, 1, 2)  # (T, 3, H, W)

        label = self.label_map[str(row["emotion"]).strip().lower()]
        return frames_t, torch.tensor(label, dtype=torch.long)

def build_dataloader(df, label_map, batch_size=8, shuffle=True, is_train=False, use_weighted_sampler=True):
    dataset = TemporalDataset(df, label_map, is_train=is_train)

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
