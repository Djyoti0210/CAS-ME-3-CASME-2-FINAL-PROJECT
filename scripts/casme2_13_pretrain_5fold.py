import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings

class CASME2CVDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        
        # Hardcode classes to ensure fixed order across folds
        self.classes = ['disgust', 'happy', 'others', 'repression', 'surprise']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        
        # We need to make sure image exists, though prepare script extracts them all
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[row['emotion']]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_weighted_sampler_and_loss_weights(dataset):
    """
    Computes WeightedRandomSampler and class weights for loss function
    based on the current training fold dataset.
    """
    targets = [dataset.class_to_idx[emotion] for emotion in dataset.df['emotion']]
    class_counts = np.bincount(targets, minlength=len(dataset.classes))
    
    # Loss Weights
    total_samples = len(targets)
    num_classes = len(dataset.classes)
    loss_weights = []
    for count in class_counts:
        weight = total_samples / (num_classes * count) if count > 0 else 0.0
        loss_weights.append(weight)
    
    # Sampler Weights
    class_sampler_weights = 1.0 / np.maximum(class_counts, 1) # Prevent division by zero
    sample_weights = [class_sampler_weights[t] for t in targets]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler, torch.FloatTensor(loss_weights), class_counts

def build_model(num_classes):
    """Build ResNet18 (Frozen except layer4 and FC)"""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    # Replace FC layer (automatically requires_grad=True)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )
    
    return model

def main():
    print("=== CASME2 5-Fold Pretraining ===")
    
    manifest_path = r"D:\CASME PROJECT FINAL\manifest_casme2_5fold.csv"
    img_dir = r"D:\CASME PROJECT FINAL\CASME2_5fold_pool\images"
    
    batch_size = 32
    num_epochs = 100
    patience = 8
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load entire manifest
    df = pd.read_csv(manifest_path)
    print(f"Total samples: {len(df)}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Fold Results tracking
    fold_results = []
    
    for fold in range(5):
        print(f"\n=====================================")
        print(f"=== FOLD {fold} ===")
        print(f"=====================================")
        
        # Split Data
        train_df = df[df['fold'] != fold]
        val_df = df[df['fold'] == fold]
        
        train_dataset = CASME2CVDataset(train_df, img_dir, transform=train_transform)
        val_dataset = CASME2CVDataset(val_df, img_dir, transform=val_transform)
        
        classes = train_dataset.classes
        num_classes = len(classes)
        
        # Sampler and loss weights
        sampler, loss_weights, class_counts = get_weighted_sampler_and_loss_weights(train_dataset)
        loss_weights = loss_weights.to(device)
        
        print(f"Train Counts: {dict(zip(classes, class_counts))}")
        print(f"Loss Class Weights: {loss_weights.cpu().numpy()}")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Model
        model = build_model(num_classes).to(device)
        
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss(weight=loss_weights)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-3)
        
        # Training Tracking
        best_val_macro_f1 = 0.0
        best_val_acc = 0.0
        best_report = None
        epochs_without_improvement = 0
        best_weights_path = rf"D:\CASME PROJECT FINAL\casme2_pretrained_fold{fold}.pt"
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                
            train_loss = train_loss / len(train_dataset)
            
            # Validation
            model.eval()
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
                    
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                val_acc = accuracy_score(val_labels, val_preds)
                val_macro_f1 = f1_score(val_labels, val_preds, average='macro')
                
            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_macro_f1:.4f}")
            
            # Early stopping based on Val Macro-F1 (per user instruction)
            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                best_val_acc = val_acc
                epochs_without_improvement = 0
                torch.save(model.state_dict(), best_weights_path)
                
                # Pre-calculate report for the best epoch
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    best_report = classification_report(val_labels, val_preds, target_names=classes, output_dict=True, zero_division=0)
                
                print(f"  -> Best model saved! (Macro-F1 improved)")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break
        
        print(f"\nFold {fold} Best Val Acc: {best_val_acc:.4f} | Best Val Macro-F1: {best_val_macro_f1:.4f}")
        fold_results.append({
            'fold': fold,
            'val_acc': best_val_acc,
            'val_macro_f1': best_val_macro_f1,
            'report': best_report
        })
        
    print("\n=====================================")
    print("=== FINAL 5-FOLD CROSS-VALIDATION RESULTS ===")
    print("=====================================")
    
    accs = [r['val_acc'] for r in fold_results]
    f1s = [r['val_macro_f1'] for r in fold_results]
    
    print(f"Mean Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Mean Macro-F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    
    print("\nMean Per-Class F1-Scores:")
    for cls_name in classes:
        cls_f1s = [r['report'][cls_name]['f1-score'] for r in fold_results]
        print(f"{cls_name:<15}: {np.mean(cls_f1s):.4f} ± {np.std(cls_f1s):.4f}")

if __name__ == "__main__":
    main()
