import pandas as pd
import sys
import os
import subprocess

def analyze():
    df = pd.read_csv("manifest.csv")
    df["emotion"] = df["emotion"].str.strip().str.lower()
    
    total = len(df)
    print("=" * 80)
    print("GLOBAL CLASS BALANCE")
    print("=" * 80)
    counts = df["emotion"].value_counts()
    for emotion, count in counts.items():
        pct = (count / total) * 100
        warning = "  <<< WARNING: FEWER THAN 15 SAMPLES!" if count < 15 else ""
        print(f"{emotion:<15} : {count:>4} samples ({pct:>5.1f}%){warning}")
        
    print("\n" + "=" * 80)
    print("CLASS BALANCE PER FOLD")
    print("=" * 80)
    folds = sorted(df["fold"].unique())
    for fold in folds:
        print(f"\n--- FOLD {fold} ---")
        fold_df = df[df["fold"] == fold]
        fold_counts = fold_df["emotion"].value_counts()
        for emotion in counts.index:
            c = fold_counts.get(emotion, 0)
            warning = "  <<< ZERO SAMPLES IN THIS FOLD!" if c == 0 else ""
            print(f"  {emotion:<13} : {c:>3}{warning}")

    print("\n" + "=" * 80)
    print("CONFUSION MATRIX / RECALL FROM 09_compute_uf1_uar.py")
    print("=" * 80)
    try:
        # Run the existing script and capture its output
        result = subprocess.run([sys.executable, "scripts/09_compute_uf1_uar.py"], 
                                capture_output=True, text=True, check=True)
        # We just want the classification report part at the end
        out = result.stdout
        marker = "Full classification report"
        if marker in out:
            print(marker + out.split(marker)[1])
        else:
            print("Could not find classification report in output. Here is the full output:")
            print(out)
    except Exception as e:
        print(f"Error running 09_compute_uf1_uar.py: {e}")
        if hasattr(e, 'output'):
            print(e.output)
        if hasattr(e, 'stderr'):
            print(e.stderr)

if __name__ == '__main__':
    analyze()
