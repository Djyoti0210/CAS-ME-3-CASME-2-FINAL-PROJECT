"""
00b_inspect_frames.py -- Run this to see how frame/ and depth/ folders are
"""
from pathlib import Path

DATASET_ROOT = Path("CASME3_Dataset")
clip_root = DATASET_ROOT / "Part_A_ME_clip" / "Part_A_ME_clip"

frame_dir = clip_root / "frame"
depth_dir = clip_root / "depth"
video_dir = clip_root / "video"

for name, d in [("frame", frame_dir), ("depth", depth_dir), ("video", video_dir)]:
    print("=" * 60)
    print(f"{name}/ contents (first 3 levels, first few items each)")
    print("=" * 60)
    if not d.exists():
        print("  NOT FOUND at", d)
        continue
    level1 = sorted(d.iterdir())
    print(f"  Level 1 ({len(level1)} items), first 3:")
    for a in level1[:3]:
        print("   ", a.name, "(dir)" if a.is_dir() else "(file)")
        if a.is_dir():
            level2 = sorted(a.iterdir())
            print(f"      Level 2 ({len(level2)} items), first 5:")
            for b in level2[:5]:
                print("       ", b.name, "(dir)" if b.is_dir() else "(file)")
                if b.is_dir():
                    level3 = sorted(b.iterdir())
                    print(f"          Level 3 ({len(level3)} items), first 5:")
                    for c in level3[:5]:
                        print("           ", c.name)
    print()