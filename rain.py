"""
rain.py: the Rain Board. 115 years of IMD rainfall for 36 regions.

FasalScope · © 2026 Dr. Nilesh P. Ingle · MIT licence
Data: India Meteorological Department subdivision rainfall 1901–2015 (via data.gov.in).
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.models import ColorBar, ColumnDataSource, HoverTool, LinearColorMapper, Span
from bokeh.palettes import Viridis256
from bokeh.plotting import figure

DATA = Path(__file__).parent / "data"
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
DEFICIENT = 0.8   # IMD calls a year "deficient" when rain is at least 20% below normal


def load():
    rain = pd.read_csv(DATA / "rainfall_india_1901_2015.csv")
    regions = pd.read_csv(DATA / "imd_regions.csv")
    return rain, regions


def region_table(rain, regions):
    rows = []
    for sub, g in rain.groupby("SUBDIVISION"):
        a = g["ANNUAL"].dropna()
        normal = a.mean()
        rows.append(dict(SUBDIVISION=sub, mean=normal, sd=a.std(), years=len(a),
                         deficient=int((a < DEFICIENT * normal).sum())))
    t = pd.DataFrame(rows).merge(regions, on="SUBDIVISION")
    t["cv"] = 100 * t["sd"] / t["mean"]
    # latitude/longitude -> web-mercator metres, which map tiles use
    k = 6378137.0
    t["mx"] = t["lon"].map(lambda lon: k * math.radians(lon))
    t["my"] = t["lat"].map(lambda lat: k * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))
    return t.sort_values("name").reset_index(drop=True)


def map_figure(tab, selected=None, height=520):
    src = ColumnDataSource(tab.assign(
        radius=6 + 14 * np.sqrt(tab["mean"] / tab["mean"].max()),   # area ∝ rainfall
        lw=[4 if s == selected else 1 for s in tab["SUBDIVISION"]],
        lc=["#d6303c" if s == selected else "white" for s in tab["SUBDIVISION"]],
        mean_r=tab["mean"].round(0), sd_r=tab["sd"].round(0), cv_r=tab["cv"].round(0)))
    p = figure(x_axis_type="mercator", y_axis_type="mercator", height=height, sizing_mode="stretch_width",
               x_range=(7.3e6, 1.08e7), y_range=(6.0e5, 4.3e6), tools="pan,wheel_zoom,reset",
               active_scroll="wheel_zoom", title="Average annual rain (circle size) and deficient years (colour)")
    p.add_tile("OpenStreetMap Mapnik")
    cmap = LinearColorMapper(palette=Viridis256[::-1], low=0, high=float(tab["deficient"].max()))
    r = p.scatter("mx", "my", size="radius", source=src, fill_color={"field": "deficient", "transform": cmap},
                  fill_alpha=0.85, line_color="lc", line_width="lw")
    p.add_tools(HoverTool(renderers=[r], tooltips=[("Region", "@name"), ("Average rain", "@mean_r{0,0} mm"),
                                                   ("Year-to-year swing (SD)", "@sd_r{0,0} mm (@cv_r% of average)"),
                                                   ("Deficient years", "@deficient of @years")]))
    p.add_layout(ColorBar(color_mapper=cmap, title="deficient years", width=10), "right")
    p.axis.visible = False
    p.grid.visible = False
    return p


def trend_figure(rain, sub, name, years=(1901, 2015), height=320):
    g = rain[(rain.SUBDIVISION == sub) & rain.YEAR.between(*years)].sort_values("YEAR")
    normal = rain.loc[rain.SUBDIVISION == sub, "ANNUAL"].mean()
    g = g.assign(roll=g["ANNUAL"].rolling(10, center=True, min_periods=5).mean(),
                 status=np.where(g["ANNUAL"] < DEFICIENT * normal, "deficient", "normal or above"))
    src = ColumnDataSource(g)
    p = figure(height=height, sizing_mode="stretch_width", title=f"{name}: annual rainfall",
               x_axis_label="year", y_axis_label="mm", tools="pan,box_zoom,reset,save")
    p.line("YEAR", "ANNUAL", source=src, color="#9fb3c8", line_width=1.5, legend_label="each year")
    p.line("YEAR", "roll", source=src, color="#1f3b57", line_width=3, legend_label="10-year average")
    bad = ColumnDataSource(g[g.status == "deficient"])
    dots = p.scatter("YEAR", "ANNUAL", source=bad, size=8, color="#d6303c", legend_label="deficient year (< 80% of normal)")
    p.add_layout(Span(location=normal, dimension="width", line_dash="dashed", line_color="#5b7186"))
    p.add_tools(HoverTool(renderers=[dots], tooltips=[("Year", "@YEAR"), ("Rain", "@ANNUAL{0,0} mm")]))
    p.legend.location = "top_left"
    p.legend.background_fill_alpha = 0.6
    return p


def monthly_figure(rain, sub, name, height=320):
    g = rain[rain.SUBDIVISION == sub][MONTHS].mean()
    src = ColumnDataSource(dict(m=[m.title() for m in MONTHS], v=g.round(0).values,
                                c=["#2f6db3" if m in ("JUN", "JUL", "AUG", "SEP") else "#b8c7d6" for m in MONTHS]))
    p = figure(x_range=[m.title() for m in MONTHS], height=height, sizing_mode="stretch_width",
               title=f"{name}: average rain by month (monsoon in blue)", y_axis_label="mm", tools="save")
    p.vbar(x="m", top="v", width=0.8, color="c", source=src)
    p.add_tools(HoverTool(tooltips=[("Month", "@m"), ("Average", "@v{0,0} mm")]))
    p.xgrid.visible = False
    return p
