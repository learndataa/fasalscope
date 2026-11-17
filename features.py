"""
features.py: turn a banana photo into a few numbers a model can use.

FasalScope · © 2026 Dr. Nilesh P. Ingle · MIT licence
https://github.com/learndataa/fasalscope

Idea: a computer can't "see" ripeness, but it can count colours.
We keep only the coloured pixels (the banana), ignore the plain
background, and measure how much of the banana is green, yellow
or brown.
"""
import base64
import io

import numpy as np
from PIL import Image, ImageFilter, ImageOps

# The six numbers every model in FasalScope uses (same order everywhere).
FEATURES = ["pct_green", "pct_yellow", "pct_brown", "mean_hue", "mean_sat", "mean_val"]

FEATURE_LABELS = {
    "pct_green": "% green",
    "pct_yellow": "% yellow",
    "pct_brown": "% brown / dark",
    "mean_hue": "average hue (°)",
    "mean_sat": "average saturation (%)",
    "mean_val": "average brightness (%)",
}


def _load(img):
    """Accept a file path, bytes, a file-like object or a PIL image."""
    if isinstance(img, Image.Image):
        im = img
    elif isinstance(img, (bytes, bytearray)):
        im = Image.open(io.BytesIO(img))
    else:
        im = Image.open(img)
    im = ImageOps.exif_transpose(im)  # phone photos store rotation separately
    return im.convert("RGB")


def thumbnail_b64(im, size=96):
    """A tiny JPEG (about 3 KB) as text, so it can live inside the database."""
    t = im.copy()
    t.thumbnail((size, size))
    buf = io.BytesIO()
    t.save(buf, format="JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def banana_mask(hsv):
    """
    Which pixels belong to the banana?
    1. Fruit-coloured pixels: yellow-to-green hue, clearly coloured, not too dark.
       (Wooden tables, hands and white walls fall outside this range.)
    2. Dark spots: dark pixels count only if they sit on or right next to fruit-coloured pixels.
    """
    h = hsv[..., 0] * 360.0
    s, v = hsv[..., 1], hsv[..., 2]
    fruit = (h >= 40) & (h < 170) & (s > 0.28) & (v > 0.25)
    near = np.asarray(Image.fromarray((fruit * 255).astype("uint8")).filter(ImageFilter.MaxFilter(7))) > 0
    dark = (v <= 0.35) & (s > 0.15) & near
    return fruit | dark


def extract(img, max_side=256):
    """
    Return (features: dict, thumbnail: str, coverage: float).
    coverage = share of the photo that looked like fruit (0 to 1).
    """
    im = _load(img)
    small = im.copy()
    small.thumbnail((max_side, max_side))
    hsv = np.asarray(small.convert("HSV"), dtype=float) / 255.0
    h_deg = hsv[..., 0] * 360.0
    s, v = hsv[..., 1], hsv[..., 2]

    m = banana_mask(hsv)
    coverage = float(m.mean())
    if m.sum() < 50:  # almost nothing coloured: no banana found
        return None, thumbnail_b64(im), coverage

    h, sat, val = h_deg[m], s[m], v[m]
    green = (h >= 65) & (h < 170) & (val > 0.25)
    yellow = (h >= 38) & (h < 65) & (val > 0.35)
    brown = ~green & ~yellow  # dark spots and brown patches

    # Hue is an angle, so average it with a little trigonometry (0° and 359° are neighbours).
    ang = np.deg2rad(h)
    mean_hue = float(np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360)

    feats = {
        "pct_green": round(100 * float(green.mean()), 2),
        "pct_yellow": round(100 * float(yellow.mean()), 2),
        "pct_brown": round(100 * float(brown.mean()), 2),
        "mean_hue": round(mean_hue, 2),
        "mean_sat": round(100 * float(sat.mean()), 2),
        "mean_val": round(100 * float(val.mean()), 2),
    }
    return feats, thumbnail_b64(im), coverage
