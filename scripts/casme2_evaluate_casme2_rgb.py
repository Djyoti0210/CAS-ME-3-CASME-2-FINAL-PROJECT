import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score
import warnings

def main():
    data_dir = 'CASME2_static_image_pool_backup'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    holdout_dataset = datasets.ImageFolder(os.path.join(data_dir, 'holdout'), transform=transform, allow_empty=True)
    holdout_loader = DataLoader(holdout_dataset, batch_size=32, shuffle=False, num_workers=4)
    classes = holdout_dataset.classes
    
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, len(classes))
    )
    
    weights_path = 'casme2_pretrained_rgb_backbone.pt'
    model.load_state_dict(torch.load(weights_path))
    model = model.to(device)
    model.eval()
    
    final_preds = []
    final_labels = []
    
    with torch.no_grad():
        for inputs, labels in holdout_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            final_preds.extend(preds.cpu().numpy())
            final_labels.extend(labels.numpy())
            
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_acc = accuracy_score(final_labels, final_preds)
        final_macro_f1 = f1_score(final_labels, final_preds, average='macro')
        report = classification_report(final_labels, final_preds, labels=list(range(len(classes))), target_names=classes, zero_division=0)
        
    print("=== FINAL EVALUATION ON HOLDOUT ===")
    print(f"Holdout Accuracy: {final_acc:.4f}")
    print(f"Holdout Macro-F1: {final_macro_f1:.4f}")
    print("\nPer-class Report:")
    print(report)

if __name__ == '__main__':
    main()
