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

# Fix random seed for reproducibility
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
        img_path = self.img_dir / row['image_path']
        image = Image.open(img_path).convert("RGB")
        label = self.label_map[row['emotion']]
        
        if self.transform:
            image = self.transform(image)
            
        # Return clip_id for Majority Voting later
        clip_id = f"{row['subject']}/{row['clip']}"
        return image, label, clip_id

def build_model(num_classes):
    # 1. Architecture Upgrade: ConvNeXt-Tiny (Modern, extremely strong spatial priors)
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    
    # Freeze earlier stages, Unfreeze final stage (features.7) and classifier
    for name, param in model.named_parameters():
        param.requires_grad = False
        if "features.7" in name or "classifier" in name:
            param.requires_grad = True
            
    # Modify classifier to add dropout
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )
    return model

def get_transforms():
    # Slightly higher resolution crop (224 is standard, but ConvNeXt thrives on rich textures)
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

def train_one_fold(fold_idx, df, img_dir, label_map, device, is_final=False, epochs=50):
    train_transform, val_transform = get_transforms()
    
    if is_final:
        train_df = df.copy()
        val_df = None
    else:
        train_df = df[df['fold'] != fold_idx].copy()
        val_df = df[df['fold'] == fold_idx].copy()
        
    num_classes = len(label_map)
    
    # Class weights & Sampler for Train
    class_counts = train_df['emotion'].value_counts()
    total_train = len(train_df)
    weights_dict = {cls: total_train / count for cls, count in class_counts.items()}
    sample_weights = [weights_dict[e] for e in train_df['emotion']]
    
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_dataset = CASME2Dataset(train_df, img_dir, label_map, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True)
    
    if not is_final:
        val_dataset = CASME2Dataset(val_df, img_dir, label_map, transform=val_transform)
        # Validation doesn't need shuffle. Keeping batch size robust.
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        
    # Inverse frequency loss weighting + Label Smoothing
    loss_weights = torch.zeros(num_classes)
    for cls, idx in label_map.items():
        loss_weights[idx] = weights_dict[cls]
    loss_weights = loss_weights.to(device)
    
    model = build_model(num_classes).to(device)
    
    # 2. Modern Optimization: Label Smoothing (0.1)
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=0.1)
    
    # Optimize parameters
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-3)
    
    # 3. Modern Optimization: Cosine Annealing (smooth decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_f1 = 0.0
    best_metrics = {}
    patience = 8
    patience_counter = 0
    epochs_run = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping to prevent spikes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_dataset)
        scheduler.step()
        
        if is_final:
            print(f"Final Champion Model - Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f}")
            continue
            
        # --- 4. The Silver Bullet: Clip-Level Aggregation Validation ---
        model.eval()
        clip_preds_dict = collections.defaultdict(list)
        clip_labels_dict = {}
        
        with torch.no_grad():
            for inputs, labels, clips in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs) # Logits
                
                # Softmax to get probabilities before averaging
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                labels = labels.cpu().numpy()
                
                for i in range(len(clips)):
                    clip_id = clips[i]
                    clip_preds_dict[clip_id].append(probs[i])
                    clip_labels_dict[clip_id] = labels[i]
                    
        # Aggregate logic
        val_preds_clip = []
        val_labels_clip = []
        for clip_id in clip_labels_dict.keys():
            avg_probs = np.mean(clip_preds_dict[clip_id], axis=0)
            final_pred = np.argmax(avg_probs)
            
            val_preds_clip.append(final_pred)
            val_labels_clip.append(clip_labels_dict[clip_id])
                
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # All metrics are now calculated purely at the CLIP LEVEL
            val_acc = accuracy_score(val_labels_clip, val_preds_clip)
            val_f1 = f1_score(val_labels_clip, val_preds_clip, average='macro')
            report_dict = classification_report(val_labels_clip, val_preds_clip, labels=list(range(num_classes)), 
                                                target_names=list(label_map.keys()), output_dict=True, zero_division=0)
            
        print(f"Fold {fold_idx} - Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Clip-Val Acc: {val_acc:.4f} | Clip-Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_metrics = {
                'acc': val_acc,
                'f1': val_f1,
                'report': report_dict
            }
            patience_counter = 0
            epochs_run = epoch
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Fold {fold_idx} Early Stopping at epoch {epoch}")
                break
                
    if is_final:
        # Save final champion model
        save_path = "casme2_loso_backbone_champion.pt"
        torch.save(model.state_dict(), save_path)
        print(f"\n🚀 FINAL CHAMPION MODEL SAVED TO {save_path} 🚀")
        return None
        
    return best_metrics, epochs_run

def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("="*60)
    print("CHAMPION PIPELINE: ConvNeXt-Tiny + Clip-Level Aggregation")
    print("="*60)
    
    manifest_path = r"D:\CASME PROJECT FINAL\CASME2_static_image_pool\casme2_manifest.csv"
    img_dir = r"D:\CASME PROJECT FINAL\CASME2_static_image_pool\images"
    
    df = pd.read_csv(manifest_path)
    
    # Ensure 5-class taxonomy
    unique_emotions = sorted(df['emotion'].unique())
    label_map = {e: i for i, e in enumerate(unique_emotions)}
    print(f"Taxonomy ({len(label_map)} classes): {label_map}")
    
    num_folds = df['fold'].nunique()
    print(f"Starting True LOSO Evaluation over {num_folds} folds...")
    
    all_metrics = []
    epochs_list = []
    
    for fold_idx in sorted(df['fold'].unique()):
        print(f"\n{'='*40}\nProcessing Fold {fold_idx}\n{'='*40}")
        # Default epochs set higher (60) since cosine decay allows longer training
        metrics, epochs_run = train_one_fold(fold_idx, df, img_dir, label_map, device, epochs=60)
        
        # Check for erratic fold
        val_f1 = metrics['f1']
        val_acc = metrics['acc']
        if val_f1 < 0.20:
            print(f"  >>> WARNING: Fold {fold_idx} behaving erratically (Macro-F1: {val_f1:.4f})")
            
        all_metrics.append(metrics)
        epochs_list.append(epochs_run)
        
    # Summarize results
    accs = [m['acc'] for m in all_metrics]
    f1s = [m['f1'] for m in all_metrics]
    
    # Using Median to avoid early-stopping skew from 2-clip subjects
    avg_epochs = int(np.median(epochs_list))
    
    print("\n" + "="*50)
    print("=== CASME2 CHAMPION LOSO (26 FOLDS) CLIP-LEVEL RESULTS ===")
    print("="*50)
    print(f"Mean Clip Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Mean Clip Macro-F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    
    print("\n--- Per-Class Clip F1 (Averaged across folds) ---")
    for cls in label_map.keys():
        cls_f1s = [m['report'][cls]['f1-score'] for m in all_metrics]
        print(f"  {cls:<12}: {np.mean(cls_f1s):.4f} ± {np.std(cls_f1s):.4f}")
    print("="*50)
    
    print(f"\nRunning Final All-Data Pass for Champion Checkpoint (using robust median {avg_epochs} epochs)...")
    train_one_fold(-1, df, img_dir, label_map, device, is_final=True, epochs=avg_epochs)
    
if __name__ == "__main__":
    main()
