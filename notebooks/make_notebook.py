"""Generates FasalScope_under_the_hood.ipynb (run from the repo root)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip()))


md("""
# 🍌 FasalScope: under the hood
**From a photo to a decision, step by step.**
© 2026 Dr. Nilesh P. Ingle · MIT licence · [github.com/learndataa/fasalscope](https://github.com/learndataa/fasalscope)

This notebook uses the *same code* as the live app. Click **Runtime → Run all**. Nothing to install, no accounts needed.

| Part | What you'll see |
|---|---|
| 1 | Getting the code |
| 2 | A photo becomes six numbers |
| 3 | The training data, as statistics and a chart |
| 4 | Three models: SVM, Naive Bayes, logistic regression |
| 5 | The threshold dial (drag the slider) |
| 6 | Overfitting (drag the slider) |
| 7 | The Rain Board charts |
| 8 | *(Optional)* Reading live uploads from MongoDB |
""")

md("## 1 · Get the code")
code("""
import os, sys
if not os.path.exists("app.py"):
    if not os.path.exists("fasalscope"):
        !git clone -q https://github.com/learndataa/fasalscope
    os.chdir("fasalscope")
sys.path.insert(0, os.getcwd())
print("Working in", os.getcwd())
""")
code("""
import pandas as pd
import numpy as np
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
output_notebook(hide_banner=True)

import features, models, inspector, rain, community
print("Loaded FasalScope modules.")
""")

md("""
## 2 · A photo becomes six numbers
A computer can't "see" ripeness, but it can **count colours**. `features.extract` keeps only the coloured pixels (the banana), ignores the plain background, and measures:

`% green`, `% yellow`, `% brown/dark`, average hue, saturation and brightness.

Run the next cell. In Colab, you can upload your own banana photo; press Cancel to use an example photo instead.
""")
code("""
from PIL import Image
photo = None
try:
    from google.colab import files
    print("Choose a banana photo (or press Cancel to use a test banana)…")
    up = files.upload()
    photo = next(iter(up)) if up else None
except Exception:
    pass
if photo is None:
    photo = "samples/ripe_example.jpg"   # try "samples/unripe_example.jpg" too

feats, thumb, coverage = features.extract(photo)
display(Image.open(photo).resize((240, 180)))
print(f"{100*coverage:.0f}% of the photo looked like fruit")
# See which pixels counted as banana (the rest is darkened):
import numpy as np
im = Image.open(photo).convert("RGB"); im.thumbnail((256, 256))
m = features.banana_mask(np.asarray(im.convert("HSV"), float) / 255)
ov = np.asarray(im).copy(); ov[~m] = (ov[~m] * 0.25).astype("uint8")
display(Image.fromarray(ov))
pd.Series(feats, name="value")
""")

md("""
## 3 · The training data
Each row is one labelled photo from the *Ripe Unripe Banana Dataset* (Mendeley Data, CC BY 4.0), already turned into six numbers.

**Be careful with the accuracy you'll see.** The dataset has many photos of the *same* bananas on the *same* table, so the hidden test photos look a lot like the training photos. Real phone photos will be harder. This problem is called **data leakage**.
""")
code("""
df = pd.read_csv("data/banana_features.csv")
print(df.label.value_counts(), "\\n")
df.groupby("label")[features.FEATURES].mean().round(1)
""")
code("""
from bokeh.models import ColumnDataSource, HoverTool
colors = {"ripe": "#e0a020", "unripe": "#3f7d3a"}
p = figure(height=420, width=640, title="Every training photo as a dot",
           x_axis_label="% green", y_axis_label="% yellow", tools="pan,wheel_zoom,box_zoom,reset")
for lab, g in df.groupby("label"):
    p.scatter(g.pct_green, g.pct_yellow, color=colors[lab], alpha=0.5, size=6, legend_label=lab)
show(p)
""")
md("**Try:** change `pct_yellow` to `pct_brown` or `mean_val` in the cell above. Which pair of features separates the classes best?")

md("""
## 4 · Three models
`models.train` splits the data (75% to learn from, 25% hidden for testing), trains three classic models and measures each one on the hidden photos.
""")
code("""
T = models.train(df)
print(f"trained on {T.n_train} photos, tested on {T.n_test} hidden photos\\n")
for name, acc in T.accuracy.items():
    print(f"{name:22s} test accuracy {100*acc:5.1f}%")
print("\\nYour photo:", {k: f"{100*v:.0f}% ripe" for k, v in models.predict(T, feats).items()})
""")
code("""
from sklearn.metrics import confusion_matrix
for name, s in T.test_scores.items():
    tn, fp, fn, tp = confusion_matrix(T.y_test, s >= 0.5).ravel()
    print(f"{name:22s} true unripe {tn:3d} | false alarms {fp:3d} | missed {fn:3d} | true ripe {tp:3d}")
""")

md("""
### Is it even a banana?
The models can only answer *ripe or unripe*. Show them a yellow rubber duck and they say "100% ripe"!
So the app first asks a different question: **does this photo look like the training photos?**
It measures how far the photo's numbers (plus one shape number, *elongation*) are from the nearest group of training photos.
""")
code("""
unusual, score, reasons = T.familiar.check(feats)
print(f"unusualness {score:.1f}  (99.5% of training photos score below {T.familiar.threshold:.1f})")
print("⚠ doesn't look like a training banana:" if unusual else "looks like a training banana", ", ".join(reasons))
""")

md("""
## 5 · The threshold dial
Every model gives a **score**; *you* choose the threshold that turns it into a decision. Drag the slider: false alarms and misses trade places. (It's the same idea as the health-camp test in Unit IV.)
""")
code("""
print(f"Weak model (brightness only): {100*T.weak_accuracy:.0f}% right at threshold 0.5")
show(inspector.threshold_demo(T.weak_scores, T.y_test, "brightness-only model"))
""")
md("With all six numbers the problem is easy, so the dial hardly matters. Compare:")
code("""show(inspector.threshold_demo(T.test_scores["Logistic regression"], T.y_test))""")

md("""
## 6 · A model that memorises
Can we predict Vidarbha's rainfall from the year? A wigglier curve always fits the training years better. Watch what happens to the **hidden** years as you increase the degree.
""")
code("""
r, regions = rain.load()
v = r[r.SUBDIVISION == "VIDARBHA"].dropna(subset=["ANNUAL"])
layout, errors = inspector.overfit_demo(v.YEAR, v.ANNUAL)
show(layout)
""")

md("## 7 · The Rain Board")
code("""
tab = rain.region_table(r, regions)
show(rain.map_figure(tab, "VIDARBHA"))
show(rain.trend_figure(r, "VIDARBHA", "Vidarbha"))
show(rain.monthly_figure(r, "VIDARBHA", "Vidarbha"))
""")
md("**Try:** change `\"VIDARBHA\"` to `\"WEST RAJASTHAN\"` or `\"KERALA\"`. Which region has the most deficient years?")

md("""
## 8 · *(Optional)* Live uploads from MongoDB
Only if you have the connection string. In Colab: click the 🔑 **Secrets** icon (left bar), add `MONGO_URI`, and allow this notebook to read it. Without it, this cell just says so.
""")
code("""
import importlib, db
try:
    from google.colab import userdata
    os.environ["MONGO_URI"] = userdata.get("MONGO_URI")
except Exception:
    pass
store, msg = db.get_store()
print(msg)
uploads = store.all()
print(len(uploads), "community uploads")
uploads.drop(columns=["thumb"], errors="ignore").head()
""")

md("""
## Exercises
1. **Feature:** add a seventh feature to `features.py` (for example, the share of very dark pixels). Does accuracy change?
2. **Model:** add a k-nearest-neighbours model to `models.make_models()` and compare.
3. **Threshold:** for a fruit seller who hates selling unripe bananas, which threshold would you pick, and why?
4. **Rain:** which region has the biggest *relative* swing (SD ÷ average)? Use `tab.sort_values("cv")`.
""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                  "colab": {"provenance": []}}
nbf.write(nb, "notebooks/FasalScope_under_the_hood.ipynb")
print("wrote notebooks/FasalScope_under_the_hood.ipynb")
