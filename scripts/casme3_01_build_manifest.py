"""
01_build_manifest.py -- Builds a clean manifest.csv linking each annotation
row (Subject, Filename, Onset, Apex, Offset, emotion) to real onset/apex/offset
jpg + depth file paths on disk. Run this after 00b_inspect_frames.py confirms
the folder structure.
"""
from pathlib import Path
import pandas as pd

DATASET_ROOT = Path("CASME3_Dataset")
ANNOTATION_FILE = DATASET_ROOT / "PartA_annotation" / "cas(me)3_part_A_ME_label_JpgIndex_v2.xlsx"
CLIP_ROOT = DATASET_ROOT / "Part_A_ME_clip" / "Part_A_ME_clip"
FRAME_DIR = CLIP_ROOT / "frame"
DEPTH_DIR = CLIP_ROOT / "depth"

OUTPUT_MANIFEST = Path("manifest.csv")

def get_cached_dirs(base_dir: Path):
    if not base_dir.exists():
        return []
    return [d for d in base_dir.iterdir() if d.is_dir()]

def find_clip_folder(dirs: list, subject: str, filename: str, onset: int):
    """Folder names look like spNO.10_a_116 -- match exactly first, then fallback to prefix."""
    exact_name = f"{subject}_{filename}_{onset}"
    for d in dirs:
        if d.name == exact_name:
            return d
    
    prefix = f"{subject}_{filename}_"
    matches = [d for d in dirs if d.name.startswith(prefix)]
    if len(matches) == 0:
        return None
    if len(matches) > 1:
        print(f"  WARNING: multiple folders match {prefix} -- using first: {matches}")
    return matches[0]

def main():
    df = pd.read_excel(ANNOTATION_FILE)
    print(f"Loaded {len(df)} annotation rows")
    print("Columns:", list(df.columns))

    print("Caching directory lists (this should be fast)...")
    frame_dirs = get_cached_dirs(FRAME_DIR)
    depth_dirs = get_cached_dirs(DEPTH_DIR)

    rows = []
    missing = 0

    for idx, row in df.iterrows():
        subject = str(row["Subject"]).strip()
        filename = str(row["Filename"]).strip()
        onset = int(row["Onset"])
        apex = int(row["Apex"])
        offset = int(row["Offset"])
        emotion = str(row["emotion"]).strip().lower()

        frame_folder = find_clip_folder(frame_dirs, subject, filename, onset)
        depth_folder = find_clip_folder(depth_dirs, subject, filename, onset)

        if frame_folder is None or depth_folder is None:
            missing += 1
            continue

        onset_jpg = frame_folder / f"{onset}.jpg"
        apex_jpg = frame_folder / f"{apex}.jpg"
        offset_jpg = frame_folder / f"{offset}.jpg"
        onset_depth = depth_folder / f"{onset}.png"
        apex_depth = depth_folder / f"{apex}.png"
        offset_depth = depth_folder / f"{offset}.png"

        if not (onset_jpg.exists() and apex_jpg.exists()):
            missing += 1
            continue

        clip_name = f"{subject}_{filename}_{onset}"

        rows.append({
            "clip_name": clip_name,
            "subject": subject,
            "filename": filename,
            "onset": onset,
            "apex": apex,
            "offset": offset if offset_jpg.exists() else apex,  # fallback if offset frame missing
            "emotion": emotion,
            "frame_folder": str(frame_folder),
            "depth_folder": str(depth_folder),
        })

    manifest = pd.DataFrame(rows)
    print(f"\nMatched: {len(manifest)} / {len(df)}  (missing/skipped: {missing})")
    print("\nEmotion class distribution:")
    print(manifest["emotion"].value_counts())
    print("\nUnique subjects:", manifest["subject"].nunique())

    manifest.to_csv(OUTPUT_MANIFEST, index=False)
    print(f"\nSaved manifest to {OUTPUT_MANIFEST.resolve()}")

if __name__ == "__main__":
    main()