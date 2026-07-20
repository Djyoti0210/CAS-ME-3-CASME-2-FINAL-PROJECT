import pandas as pd
import numpy as np
from pathlib import Path

metrics_dir = Path('metrics')
results = {}

for i in range(5):
    df = pd.read_csv(metrics_dir / f'exp_f_fold_{i}_metrics.csv')
    best_idx = df['val_uf1'].idxmax()
    results[i] = df.loc[best_idx]
    print(f"Fold {i}: UF1={results[i]['val_uf1']:.4f}, UAR={results[i]['val_uar']:.4f}")

uf1s = [r['val_uf1'] for r in results.values()]
uars = [r['val_uar'] for r in results.values()]

print("-" * 40)
print(f"Mean UF1: {np.mean(uf1s):.4f} ± {np.std(uf1s):.4f}")
print(f"Mean UAR: {np.mean(uars):.4f} ± {np.std(uars):.4f}")
