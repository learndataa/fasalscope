"""
build_dataset.py: turn a folder of labelled banana photos into data/banana_features.csv.

FasalScope · © 2026 Dr. Nilesh P. Ingle · MIT licence

Usage:  python build_dataset.py PATH/TO/PHOTOS
Labels come from file names (Raw_…/Unripe_… = unripe, Ripe_… = ripe) or, failing that,
from folder names, e.g. PHOTOS/Ripe/*.jpg and PHOTOS/Unripe/*.jpg.
"""
import sys
from pathlib import Path

import pandas as pd

from features import FEATURES, SHAPE, extract

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def label_for(path: Path):
    """Label from the file name first (Mendeley: Raw_Banana… / Ripe_Banana…), then from folder names."""
    for part in [path.name.lower()] + [p.lower() for p in reversed(path.parts[:-1])]:
        if "unripe" in part or part.startswith("raw"):
            return "unripe"
        if "ripe" in part:
            return "ripe"
    return None


def main(root, out="data/banana_features.csv", source="Mendeley: Ripe Unripe Banana Dataset (CC BY 4.0)"):
    rows, skipped = [], 0
    for f in sorted(Path(root).rglob("*")):
        if f.suffix.lower() not in EXT:
            continue
        lab = label_for(f.relative_to(root))
        feats = extract(f)[0] if lab else None
        if feats is None:
            skipped += 1
            continue
        rows.append({**feats, "label": lab, "file": f.name, "source": source})
    df = pd.DataFrame(rows, columns=FEATURES + [SHAPE, "label", "file", "source"])
    df.to_csv(out, index=False)
    print(f"{len(df)} photos -> {out} ({skipped} skipped)\n{df.label.value_counts().to_string()}")


if __name__ == "__main__":
    main(sys.argv[1])
