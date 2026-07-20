"""
01b_add_folds.py -- Adds a subject-independent stratified 5-fold 'fold' column
to manifest.csv. Ensures no subject appears in both train and val for any fold.
Run once, right after 01_build_manifest.py.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

MANIFEST_PATH = Path("manifest.csv")
N_FOLDS = 5
SEED = 42

def main():
    df = pd.read_csv(MANIFEST_PATH)
    print(f"Loaded manifest: {len(df)} rows, {df['subject'].nunique()} subjects")

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    df["fold"] = -1
    X = df.index.values
    y = df["emotion"].values
    groups = df["subject"].values

    for fold_idx, (_, val_idx) in enumerate(sgkf.split(X, y, groups)):
        df.loc[val_idx, "fold"] = fold_idx

    assert (df["fold"] == -1).sum() == 0, "Some rows did not get a fold assigned!"

    print("\nFold sizes:")
    print(df["fold"].value_counts().sort_index())

    print("\nChecking no subject leaks across folds...")
    leak_found = False
    for s in df["subject"].unique():
        folds_for_subject = df[df["subject"] == s]["fold"].unique()
        if len(folds_for_subject) > 1:
            print(f"  LEAK: subject {s} appears in folds {folds_for_subject}")
            leak_found = True
    if not leak_found:
        print("  OK -- every subject is contained within a single fold.")

    print("\nPer-fold class distribution:")
    print(pd.crosstab(df["fold"], df["emotion"]))

    df.to_csv(MANIFEST_PATH, index=False)
    print(f"\nUpdated manifest saved to {MANIFEST_PATH.resolve()}")

if __name__ == "__main__":
    main()
