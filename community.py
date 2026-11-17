"""
community.py: the live dashboard of everyone's uploads (Bokeh).

FasalScope · © 2026 Dr. Nilesh P. Ingle · MIT licence
"""
import numpy as np
import pandas as pd
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, DataTable, HoverTool, TableColumn
from bokeh.plotting import figure

from features import FEATURE_LABELS
from models import PLOT_X, PLOT_Y

COL = {"ripe": "#e0a020", "unripe": "#3f7d3a"}

TIP = """
<div style="width:150px">
  <img src="@thumb" style="width:96px;border-radius:6px"><br>
  <b>@name</b><br>model says: <b>@verdict</b><br>@note<br>
  @{x}{0.0} · @{y}{0.0}
</div>"""


def dashboard(seed: pd.DataFrame, uploads: pd.DataFrame, road2d, height=430):
    p = figure(height=height, sizing_mode="stretch_width", tools="pan,wheel_zoom,box_select,tap,reset,save",
               title="Every banana as a dot",
               x_axis_label=FEATURE_LABELS[PLOT_X], y_axis_label=FEATURE_LABELS[PLOT_Y])

    # training photos (small, faint): the "map" the models learned from
    for lab in ("unripe", "ripe"):
        d = seed[seed.label == lab]
        p.scatter(d[PLOT_X], d[PLOT_Y], size=5, color=COL[lab], alpha=0.25, legend_label=f"{lab} photo")

    # the SVM road, drawn from w1*x + w2*y + b = 0
    w1, w2, b = road2d
    xs = np.linspace(0, 100, 50)
    if abs(w2) > 1e-9:
        for c, dash in ((0, "solid"), (-1, "dashed"), (1, "dashed")):
            kw = {"legend_label": "SVM road"} if c == 0 else {}
            p.line(xs, (c - b - w1 * xs) / w2, color="#1f3b57", line_width=3 if c == 0 else 1.5, line_dash=dash, **kw)
    p.x_range.start, p.x_range.end = -2, 102
    p.y_range.start, p.y_range.end = -2, 102

    up = uploads.copy() if len(uploads) else pd.DataFrame(columns=["name", "verdict", PLOT_X, PLOT_Y, "thumb", "created_at", "unusual"])
    up = up.assign(x=up.get(PLOT_X, []), y=up.get(PLOT_Y, []), color=[COL.get(v, "#888") for v in up.get("verdict", [])])
    up = up.assign(note=["⚠ unlike the training photos" if u is True else "" for u in up.get("unusual", [False] * len(up))])
    src = ColumnDataSource(up[["name", "verdict", "x", "y", "thumb", "color", "created_at", "note"]])
    if len(up):  # (an empty colour column confuses Bokeh, so only draw stars when there are some)
        r = p.scatter("x", "y", source=src, size=16, fill_color="color", line_color="#d6303c", line_width=2.5,
                      marker="star", legend_label="upload")
        p.add_tools(HoverTool(renderers=[r], tooltips=TIP))
    p.legend.click_policy = "hide"
    p.legend.orientation = "horizontal"
    p.legend.label_text_font_size = "11px"
    p.legend.spacing = 2
    p.add_layout(p.legend[0], "below")   # below the chart, so it never hides dots on a phone

    # linked views: the table and bars share `src`, so a selection shows up in all of them
    table = DataTable(source=src, height=240, sizing_mode="stretch_width", index_position=None,
                      columns=[TableColumn(field="name", title="who"), TableColumn(field="verdict", title="verdict"),
                               TableColumn(field="created_at", title="when (UTC)")])

    counts = up["verdict"].value_counts().reindex(["unripe", "ripe"]).fillna(0) if len(up) else pd.Series([0, 0], ["unripe", "ripe"])
    bars = figure(x_range=["unripe", "ripe"], height=220, sizing_mode="stretch_width", tools="",
                  toolbar_location=None, title="Community verdicts so far")
    bars.vbar(x=["unripe", "ripe"], top=counts.values, width=0.6, color=[COL["unripe"], COL["ripe"]])
    bars.y_range.start = 0
    # stacked (not side by side) so it reads well on a phone
    return column(p, bars, table, sizing_mode="stretch_width")
