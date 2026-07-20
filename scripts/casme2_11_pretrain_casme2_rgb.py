import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import collections
import warnings

def build_model(num_classes):
    """Build ResNet18 (Fully Unfrozen) with stronger dropout"""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Replace FC layer, adding stronger dropout to combat identity overfitting
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
    
    return model

def main():
    print("=== CASME2 RGB Pretraining (Aggressive Regularization) ===")
    
    data_dir = r"D:\CASME PROJECT FINAL\CASME2_static_image_pool_backup"
    batch_size = 64  # Increased batch size for smoother gradients
    num_epochs = 100
    patience = 20
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup Datasets with aggressive augmentations to prevent facial identity memorization
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)) # Randomly hide parts of the face
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    holdout_dataset = datasets.ImageFolder(os.path.join(data_dir, 'holdout'), transform=val_transform, allow_empty=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    holdout_loader = DataLoader(holdout_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    classes = train_dataset.classes
    num_classes = len(classes)
    print(f"Classes: {classes}")
    
    class_counts = collections.Counter(train_dataset.targets)
    print(f"Train Counts: {dict(class_counts)}")
    
    # Build Model
    model = build_model(num_classes).to(device)
    
    # Verify trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable_params:,}")
    
    # Loss with Label Smoothing to prevent overconfidence and overfitting
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer with higher weight decay
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    
    # Learning Rate Scheduler to drop LR when plateauing
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # Training Loop
    best_val_acc = 0.0
    epochs_without_improvement = 0
    best_weights_path = r"D:\CASME PROJECT FINAL\casme2_pretrained_rgb_backbone.pt"
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        train_loss = train_loss / len(train_dataset)
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, labels in holdout_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            macro_f1 = f1_score(val_labels, val_preds, average='macro')
            val_acc = accuracy_score(val_labels, val_preds)
            
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {macro_f1:.4f}")
        
        # Step scheduler based on validation accuracy
        scheduler.step(val_acc)
        
        # Best Model Check (No Early Stopping, runs all 100 epochs)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_weights_path)
            print(f"  -> Best model saved! (Accuracy improved)")
                
    print("\n=== FINAL EVALUATION ON HOLDOUT ===")
    model.load_state_dict(torch.load(best_weights_path, weights_only=True))
    model.eval()
    
    final_preds = []
    final_labels = []
    
    with torch.no_grad():
        for inputs, labels in holdout_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            final_preds.extend(preds.cpu().numpy())
            final_labels.extend(labels.cpu().numpy())
            
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_acc = accuracy_score(final_labels, final_preds)
        final_macro_f1 = f1_score(final_labels, final_preds, average='macro')
        report = classification_report(final_labels, final_preds, labels=list(range(len(classes))), target_names=classes, zero_division=0)
        
    print(f"Holdout Accuracy: {final_acc:.4f}")
    print(f"Holdout Macro-F1: {final_macro_f1:.4f}")
    print("\nPer-Class Metrics:")
    print(report)
    print(f"\nModel weights saved to {best_weights_path}")

if __name__ == "__main__":
    main()

