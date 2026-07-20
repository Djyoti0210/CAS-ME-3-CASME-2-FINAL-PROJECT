import os
import collections
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
from pathlib import Path

def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class CASME2Dataset(Dataset):
    def __init__(self, df, img_dir, label_map, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row['filename']
        image = Image.open(img_path).convert("RGB")
        label = self.label_map[row['emotion']]
        
        if self.transform:
            image = self.transform(image)
            
        clip_id = row['clip_id']
        return image, label, clip_id

def build_model(num_classes):
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    
    # Freeze everything except the final stage and classifier
    for name, param in model.named_parameters():
        param.requires_grad = False
        if "features.7" in name or "classifier" in name:
            param.requires_grad = True
            
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )
    return model

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def train_one_fold(fold_idx, df, img_dir, label_map, device, is_final=False, epochs=60):
    train_transform, val_transform = get_transforms()
    
    if is_final:
        train_df = df.copy()
        val_df = None
    else:
        train_df = df[df['fold'] != fold_idx].copy()
        val_df = df[df['fold'] == fold_idx].copy()
        
    num_classes = len(label_map)
    
    class_counts = train_df['emotion'].value_counts()
    total_train = len(train_df)
    weights_dict = {cls: total_train / count for cls, count in class_counts.items()}
    sample_weights = [weights_dict[e] for e in train_df['emotion']]
    
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_dataset = CASME2Dataset(train_df, img_dir, label_map, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True)
    
    if not is_final:
        val_dataset = CASME2Dataset(val_df, img_dir, label_map, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        
    loss_weights = torch.zeros(num_classes)
    for cls, idx in label_map.items():
        loss_weights[idx] = weights_dict[cls]
    loss_weights = loss_weights.to(device)
    
    model = build_model(num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_f1 = 0.0
    best_metrics = {}
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_dataset)
        scheduler.step()
        
        if is_final:
            print(f"Final 5-Fold Champion Model - Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f}")
            continue
            
        model.eval()
        clip_preds_dict = collections.defaultdict(list)
        clip_labels_dict = {}
        
        with torch.no_grad():
            for inputs, labels, clips in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs) 
                
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                labels = labels.cpu().numpy()
                
                for i in range(len(clips)):
                    clip_id = clips[i]
                    clip_preds_dict[clip_id].append(probs[i])
                    clip_labels_dict[clip_id] = labels[i]
                    
        val_preds_clip = []
        val_labels_clip = []
        for clip_id in clip_labels_dict.keys():
            avg_probs = np.mean(clip_preds_dict[clip_id], axis=0)
            final_pred = np.argmax(avg_probs)
            
            val_preds_clip.append(final_pred)
            val_labels_clip.append(clip_labels_dict[clip_id])
                
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val_acc = accuracy_score(val_labels_clip, val_preds_clip)
            val_f1 = f1_score(val_labels_clip, val_preds_clip, average='macro')
            report_dict = classification_report(val_labels_clip, val_preds_clip, labels=list(range(num_classes)), 
                                                target_names=list(label_map.keys()), output_dict=True, zero_division=0)
            
        print(f"Fold {fold_idx} - Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Clip-Val Acc: {val_acc:.4f} | Clip-Val F1: {val_f1:.4f}")
        
        # In 5-fold, we save the best metrics across the 60 epochs without early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_metrics = {
                'acc': val_acc,
                'f1': val_f1,
                'report': report_dict
            }
                
    if is_final:
        save_path = "casme2_5fold_backbone_champion.pt"
        torch.save(model.state_dict(), save_path)
        print(f"\n🚀 FINAL 5-FOLD CHAMPION MODEL SAVED TO {save_path} 🚀")
        return None
        
    return best_metrics

def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("="*60)
    print("5-FOLD CHAMPION PIPELINE: ConvNeXt-Tiny + Clip-Level Aggregation")
    print("="*60)
    
    manifest_path = r"D:\CASME PROJECT FINAL\manifest_casme2_5fold.csv"
    img_dir = r"D:\CASME PROJECT FINAL\CASME2_5fold_pool\images"
    
    df = pd.read_csv(manifest_path)
    
    unique_emotions = sorted(df['emotion'].unique())
    label_map = {e: i for i, e in enumerate(unique_emotions)}
    print(f"Taxonomy ({len(label_map)} classes): {label_map}")
    
    num_folds = df['fold'].nunique()
    print(f"Starting Grouped 5-Fold Evaluation...")
    
    all_metrics = []
    
    for fold_idx in sorted(df['fold'].unique()):
        print(f"\n{'='*40}\nProcessing Fold {fold_idx}\n{'='*40}")
        metrics = train_one_fold(fold_idx, df, img_dir, label_map, device, epochs=60)
        all_metrics.append(metrics)
        
    accs = [m['acc'] for m in all_metrics]
    f1s = [m['f1'] for m in all_metrics]
    
    print("\n" + "="*50)
    print("=== CASME2 CHAMPION 5-FOLD CLIP-LEVEL RESULTS ===")
    print("="*50)
    print(f"Mean Clip Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Mean Clip Macro-F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    
    print("\n--- Per-Class Clip F1 (Averaged across 5 folds) ---")
    for cls in label_map.keys():
        cls_f1s = [m['report'][cls]['f1-score'] for m in all_metrics]
        print(f"  {cls:<12}: {np.mean(cls_f1s):.4f} ± {np.std(cls_f1s):.4f}")
    print("="*50)
    
    print(f"\nRunning Final All-Data Pass for Champion Checkpoint (60 epochs)...")
    train_one_fold(-1, df, img_dir, label_map, device, is_final=True, epochs=60)
    
if __name__ == "__main__":
    main()
