"""
dataset.py -- Stage 3b: three streams (RGB, Flow, Depth) from cached .npz.
Run 02c_add_depth.py before using this.
"""
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

PROCESSED_DIR = Path("processed")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def fit_label_map(emotions):
    unique = sorted(emotions.unique())
    return {e: i for i, e in enumerate(unique)}

class ThreeStreamDataset(Dataset):
    def __init__(self, df, label_map, is_train=False):
        self.df = df.reset_index(drop=True)
        self.label_map = label_map
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npz_path = PROCESSED_DIR / f"{row['clip_name']}.npz"
        data = np.load(npz_path)

        rgb = data["rgb"].copy().astype(np.float32) / 255.0        # (H, W, 3)
        flow = data["flow"].copy()                                  # (H, W, 2)
        flow = np.clip(flow / 15.0, -1.0, 1.0)
        depth = data["depth"].copy().astype(np.float32) / 255.0     # (H, W)
        depth = depth[..., None]                                     # (H, W, 1)

        if self.is_train:
            if random.random() > 0.5:
                rgb = np.flip(rgb, axis=1).copy()
                flow = np.flip(flow, axis=1).copy()
                flow[..., 0] = -flow[..., 0]
                depth = np.flip(depth, axis=1).copy()

        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD

        rgb_t = torch.from_numpy(rgb).float().permute(2, 0, 1)      # (3, H, W)
        flow_t = torch.from_numpy(flow).float().permute(2, 0, 1)    # (2, H, W)
        depth_t = torch.from_numpy(depth).float().permute(2, 0, 1)  # (1, H, W)

        label = self.label_map[str(row["emotion"]).strip().lower()]
        return rgb_t, flow_t, depth_t, torch.tensor(label, dtype=torch.long)

def build_dataloader(df, label_map, batch_size=16, shuffle=True, is_train=False, use_weighted_sampler=True):
    dataset = ThreeStreamDataset(df, label_map, is_train=is_train)

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