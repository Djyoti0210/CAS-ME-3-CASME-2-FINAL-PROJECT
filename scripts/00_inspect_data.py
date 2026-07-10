"""
00_inspect_data.py -- Run this FIRST. It just looks at what you actually have
on disk so we can build the manifest correctly. Paste output back before
moving to the next step.
"""
from pathlib import Path
import pandas as pd

# ---- EDIT THIS to point at your dataset root ----
DATASET_ROOT = Path(r"d:\CASME PROJECT FINAL\CASME3_Dataset")

annotation_dir = DATASET_ROOT / "annotation"
parta_annotation_dir = DATASET_ROOT / "PartA_annotation"
clips_dir = DATASET_ROOT / "Part_A_ME_clip"

print("=" * 60)
print("1. Top-level contents of annotation/")
print("=" * 60)
if annotation_dir.exists():
    for f in sorted(annotation_dir.iterdir())[:20]:
        print(" ", f.name)
else:
    print("  (folder not found)")

print("\n" + "=" * 60)
print("2. Top-level contents of PartA_annotation/")
print("=" * 60)
if parta_annotation_dir.exists():
    for f in sorted(parta_annotation_dir.iterdir())[:20]:
        print(" ", f.name)
else:
    print("  (folder not found)")

print("\n" + "=" * 60)
print("3. Structure of Part_A_ME_clip/ (first 2 subjects)")
print("=" * 60)
if clips_dir.exists():
    subjects = sorted([d for d in clips_dir.iterdir() if d.is_dir()])
    print(f"  Total subject folders: {len(subjects)}")
    for s in subjects[:2]:
        print(f"\n  Subject: {s.name}")
        for item in sorted(s.iterdir())[:10]:
            print("   ", item.name, "(dir)" if item.is_dir() else "(file)")
else:
    print("  (folder not found)")

print("\n" + "=" * 60)
print("4. Look for a CSV/Excel annotation file and print its columns")
print("=" * 60)
for search_dir in [annotation_dir, parta_annotation_dir]:
    if not search_dir.exists():
        continue
    for f in search_dir.rglob("*"):
        if f.suffix.lower() in [".csv", ".xlsx", ".xls"]:
            print(f"\n  Found: {f}")
            try:
                if f.suffix.lower() == ".csv":
                    df = pd.read_csv(f, nrows=5)
                else:
                    df = pd.read_excel(f, nrows=5)
                print("  Columns:", list(df.columns))
                print(df.head(3).to_string())
            except Exception as e:
                print("  Could not read:", e)
