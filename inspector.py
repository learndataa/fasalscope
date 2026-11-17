"""
inspector.py: two machine-learning ideas you can drag with a slider.

FasalScope · © 2026 Dr. Nilesh P. Ingle · MIT licence

Both demos are pure Bokeh: the slider runs a little JavaScript (CustomJS)
inside the browser, so they also work in a saved HTML file with no server.
"""
import numpy as np
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CustomJS, Div, LabelSet, Slider, Span
from bokeh.plotting import figure
from sklearn.metrics import roc_curve


# ----------------------------------------------------------------- 1. the threshold dial
def threshold_demo(scores, y, model_name="Logistic regression", start=0.5):
    scores, y = np.asarray(scores, float), np.asarray(y, int)
    rng = np.random.default_rng(0)
    pts = ColumnDataSource(dict(s=scores, yy=y + rng.uniform(-0.25, 0.25, len(y)),
                                c=np.where(y == 1, "#e0a020", "#3f7d3a"),
                                lab=np.where(y == 1, "really ripe", "really unripe")))
    strip = figure(height=230, sizing_mode="stretch_width", x_range=(-0.02, 1.02), y_range=(-0.6, 1.6),
                   title=f"Each dot is a test banana, placed by its {model_name} score P(ripe)",
                   x_axis_label="model's score: probability of ripe", tools="")
    strip.scatter("s", "yy", source=pts, color="c", size=8, alpha=0.8, legend_field="lab")
    strip.yaxis.ticker, strip.yaxis.major_label_overrides = [0, 1], {0: "unripe", 1: "ripe"}
    line = Span(location=start, dimension="height", line_color="#d6303c", line_width=3)
    strip.add_layout(line)
    strip.legend.location = "top_left"
    strip.legend.orientation = "horizontal"

    cm = ColumnDataSource(dict(x=[0, 1, 0, 1], y=[1, 1, 0, 0], n=[0, 0, 0, 0],
                               t=["", "", "", ""], col=["#cfe6c7", "#f6c9c4", "#f6c9c4", "#cfe6c7"],
                               name=["true unripe", "false alarm", "missed", "true ripe"]))
    box = figure(height=260, width=320, x_range=(-0.5, 1.5), y_range=(-0.5, 1.5), tools="",
                 title="Confusion matrix", toolbar_location=None)
    box.rect("x", "y", 0.96, 0.96, source=cm, color="col")
    box.add_layout(LabelSet(x="x", y="y", text="t", source=cm, text_align="center", text_baseline="middle",
                            text_font_size="13px"))
    box.xaxis.ticker, box.xaxis.major_label_overrides = [0, 1], {0: "model says unripe", 1: "model says ripe"}
    box.yaxis.ticker, box.yaxis.major_label_overrides = [0, 1], {0: "really ripe", 1: "really unripe"}
    box.grid.visible = False

    fpr, tpr, thr = roc_curve(y, scores)
    roc = figure(height=260, width=300, x_range=(0, 1), y_range=(0, 1.02), tools="", toolbar_location=None,
                 title="ROC curve", x_axis_label="false-alarm rate", y_axis_label="ripe bananas caught")
    roc.line(fpr, tpr, line_width=2, color="#1f3b57")
    roc.line([0, 1], [0, 1], line_dash="dashed", color="#9fb3c8")
    dot = ColumnDataSource(dict(x=[0], y=[0]))
    roc.scatter("x", "y", source=dot, size=14, color="#d6303c")

    stats = Div(text="", styles={"font-size": "15px"})
    slider = Slider(start=0.0, end=1.0, value=start, step=0.01, title="Threshold: call it ripe if score ≥",
                    sizing_mode="stretch_width")
    cb = CustomJS(args=dict(pts=pts, line=line, cm=cm, dot=dot, stats=stats, slider=slider), code="""
        const t = slider.value, s = pts.data.s, c = pts.data.c;
        let tp = 0, fp = 0, tn = 0, fn = 0;
        for (let i = 0; i < s.length; i++) {
            const ripe = c[i] === '#e0a020', said = s[i] >= t;
            if (ripe && said) tp++; else if (!ripe && said) fp++; else if (!ripe && !said) tn++; else fn++;
        }
        line.location = t;
        const n = [tn, fp, fn, tp], names = cm.data.name;
        cm.data = Object.assign({}, cm.data, {n: n, t: n.map((v, i) => v + '\\n' + names[i])});
        const P = tp + fn, N = tn + fp;
        dot.data = {x: [N ? fp / N : 0], y: [P ? tp / P : 0]};
        const acc = (tp + tn) / (P + N), prec = (tp + fp) ? tp / (tp + fp) : 1, rec = P ? tp / P : 0;
        stats.text = `<b>Accuracy</b> ${(100*acc).toFixed(0)}% &nbsp;·&nbsp; <b>Precision</b> ${(100*prec).toFixed(0)}% ` +
                     `(of bananas called ripe, how many were) &nbsp;·&nbsp; <b>Recall</b> ${(100*rec).toFixed(0)}% (of ripe bananas, how many we caught)`;
    """)
    slider.js_on_change("value", cb)
    # same calculation in Python, so the first view is already filled in
    said = scores >= start
    tp, fp = int(((y == 1) & said).sum()), int(((y == 0) & said).sum())
    tn, fn = int(((y == 0) & ~said).sum()), int(((y == 1) & ~said).sum())
    n = [tn, fp, fn, tp]
    cm.data.update(n=n, t=[f"{v}\n{nm}" for v, nm in zip(n, cm.data["name"])])
    P, N = tp + fn, tn + fp
    dot.data.update(x=[fp / N if N else 0], y=[tp / P if P else 0])
    acc, prec, rec = (tp + tn) / (P + N), (tp / (tp + fp) if tp + fp else 1), (tp / P if P else 0)
    stats.text = (f"<b>Accuracy</b> {100*acc:.0f}% &nbsp;·&nbsp; <b>Precision</b> {100*prec:.0f}% "
                  f"(of bananas called ripe, how many were) &nbsp;·&nbsp; <b>Recall</b> {100*rec:.0f}% "
                  f"(of ripe bananas, how many we caught)")
    return column(slider, stats, strip, row(box, roc), sizing_mode="stretch_width")


# ----------------------------------------------------------------- 2. overfitting
def overfit_demo(years, rain, name="Vidarbha", max_degree=12, seed=3):
    years, rain = np.asarray(years, float), np.asarray(rain, float)
    x = (years - years.mean()) / (years.max() - years.mean())   # scale to about -1..1
    rng = np.random.default_rng(seed)
    test = rng.random(len(x)) < 0.3
    grid = np.linspace(x.min(), x.max(), 300)
    gyears = grid * (years.max() - years.mean()) + years.mean()
    curves, tr_err, te_err = [], [], []
    for d in range(1, max_degree + 1):
        c = np.polyfit(x[~test], rain[~test], d)
        curves.append(np.polyval(c, grid).clip(rain.min() - 400, rain.max() + 400).tolist())
        tr_err.append(float(np.sqrt(np.mean((np.polyval(c, x[~test]) - rain[~test]) ** 2))))
        te_err.append(float(np.sqrt(np.mean((np.polyval(c, x[test]) - rain[test]) ** 2))))

    p = figure(height=300, sizing_mode="stretch_width", tools="", title=f"{name}: can we predict rain from the year?",
               x_axis_label="year", y_axis_label="annual rain (mm)", y_range=(rain.min() - 250, rain.max() + 250))
    p.scatter(years[~test], rain[~test], size=7, color="#2f6db3", legend_label="training years (model sees these)")
    p.scatter(years[test], rain[test], size=8, color="#e8735a", marker="triangle", legend_label="test years (hidden)")
    curve = ColumnDataSource(dict(x=gyears, y=curves[0]))
    p.line("x", "y", source=curve, line_width=3, color="#1f3b57")
    p.legend.location = "top_left"
    p.legend.orientation = "horizontal"
    p.legend.background_fill_alpha = 0.6

    degs = list(range(1, max_degree + 1))
    e = figure(height=300, sizing_mode="stretch_width", tools="", title="Error: training vs hidden test years",
               x_axis_label="model complexity (polynomial degree)", y_axis_label="typical error (mm)")
    e.line(degs, tr_err, line_width=2, color="#2f6db3", legend_label="training error")
    e.line(degs, te_err, line_width=2, color="#e8735a", legend_label="test error")
    mark = ColumnDataSource(dict(x=[1, 1], y=[tr_err[0], te_err[0]], c=["#2f6db3", "#e8735a"]))
    e.scatter("x", "y", source=mark, size=13, color="c")
    e.legend.location = "top_left"

    info = Div(text="", styles={"font-size": "15px"})
    slider = Slider(start=1, end=max_degree, value=1, step=1, title="Model complexity (degree)", sizing_mode="stretch_width")
    cb = CustomJS(args=dict(curve=curve, mark=mark, slider=slider, info=info, curves=curves,
                            tr=tr_err, te=te_err), code="""
        const d = slider.value, i = d - 1;
        curve.data = {x: curve.data.x, y: curves[i]};
        mark.data = {x: [d, d], y: [tr[i], te[i]], c: mark.data.c};
        const best = Math.min(...te);
        const msg = d <= 2 ? 'A simple curve: training and test errors are close.'
                  : te[i] > best + 10 ? 'Training error keeps falling, but test error rises: the curve is memorising noise (overfitting).'
                  : 'More wiggles fit the training years a little better; test error barely moves.';
        info.text = `<b>Degree ${d}</b>: training error ${tr[i].toFixed(0)} mm · test error ${te[i].toFixed(0)} mm. ${msg}`;
    """)
    slider.js_on_change("value", cb)
    info.text = (f"<b>Degree 1</b>: training error {tr_err[0]:.0f} mm · test error {te_err[0]:.0f} mm. "
                 "A simple curve: training and test errors are close.")
    layout = column(slider, info, row(p, e, sizing_mode="stretch_width"), sizing_mode="stretch_width")
    return layout, dict(train=tr_err, test=te_err)
