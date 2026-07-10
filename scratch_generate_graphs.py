import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ARTIFACTS_DIR = Path(r"C:\Users\sanch\.gemini\antigravity-ide\brain\947c132a-f711-4eae-8af9-db1d39be7e50\artifacts")
class_names = ['anger', 'disgust', 'fear', 'happy', 'others', 'sad', 'surprise']

def generate_graphs():
    cm = np.array([
        [9, 5, 5, 7, 25, 6, 7],
        [14, 149, 14, 24, 25, 8, 16],
        [3, 31, 9, 10, 14, 5, 13],
        [5, 8, 4, 11, 17, 6, 4],
        [18, 10, 17, 25, 62, 14, 12],
        [8, 9, 3, 6, 8, 16, 7],
        [4, 16, 6, 16, 18, 11, 116]
    ])

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Three-Stream + Focal Loss Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "threestream_cm.png")
    plt.close()

    recalls = [14, 61, 8, 18, 49, 28, 63]
    supports = [64, 250, 85, 55, 158, 57, 187]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax2 = ax1.twinx()
    ax1.bar(class_names, supports, color='lightgray', alpha=0.7, label='Support (Count)')
    ax2.plot(class_names, recalls, color='red', marker='o', linewidth=2, markersize=8, label='Recall (%)')
    
    ax1.set_xlabel('Emotion Class')
    ax1.set_ylabel('Number of Samples (Support)')
    ax2.set_ylabel('Recall Percentage (%)')
    ax1.set_title('Class Imbalance vs Model Recall (Three-Stream + Focal)')
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "threestream_recall_vs_support.png")
    plt.close()
    print("Graphs generated successfully.")

if __name__ == "__main__":
    generate_graphs()
