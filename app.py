"""
FasalScope: a small data-science toolkit for farmers and students.

  Tab 1  Check a banana      photo -> colour numbers -> 3 models -> ripe or unripe
  Tab 2  Community dashboard everyone's uploads, live (Bokeh + MongoDB)
  Tab 3  Rain Board          115 years of IMD rainfall, with a map
  Tab 4  Model Inspector     the threshold dial and overfitting, as sliders

© 2026 Dr. Nilesh P. Ingle · MIT licence · https://github.com/learndataa/fasalscope
Run locally:  streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_bokeh import streamlit_bokeh

import community
import inspector
import models
import rain
from db import get_store, make_record
from features import FEATURES, FEATURE_LABELS, SHAPE, extract

DATA = Path(__file__).parent / "data"
CREDIT = ("Built by **Dr. Nilesh P. Ingle** · [LinkedIn](https://www.linkedin.com/in/nileshingle/) · "
          "[Code on GitHub](https://github.com/learndataa/fasalscope) · MIT licence")

st.set_page_config(page_title="FasalScope", page_icon="🍌", layout="wide")


# ---------------------------------------------------------------- cached resources
@st.cache_data
def load_seed():
    return pd.read_csv(DATA / "banana_features.csv")


@st.cache_resource
def trained_models():
    return models.train(load_seed())


@st.cache_resource
def store():
    try:
        secrets = st.secrets
    except Exception:
        secrets = None
    return get_store(secrets)


@st.cache_data
def rain_data():
    r, regions = rain.load()
    return r, rain.region_table(r, regions)


def show(fig, key):
    streamlit_bokeh(fig, use_container_width=True, theme=None, key=key)


T = trained_models()
STORE, STORE_MSG = store()

# ---------------------------------------------------------------- header
st.title("🍌 FasalScope")
st.caption("Photo → numbers → models → decision. A learning project: not a certified agricultural tool.")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Check a banana", "Community dashboard", "Rain Board", "Model Inspector", "About"])

# ---------------------------------------------------------------- 1. check a banana
with tab1:
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.subheader("1 · Take or upload a photo")
        st.write("One banana, on a plain light background, in daylight. **Fruit only, no people.**")
        src = st.radio("Photo source", ["Upload a file", "Use the camera", "Try an example"], horizontal=True,
                       label_visibility="collapsed")
        if src == "Upload a file":
            photo = st.file_uploader("Banana photo", type=["jpg", "jpeg", "png", "webp"])
        elif src == "Use the camera":
            photo = st.camera_input("Point at the banana")
        else:
            ex = st.selectbox("Example photo (from the training dataset)", ["ripe_example.jpg", "unripe_example.jpg"])
            photo = str(Path(__file__).parent / "samples" / ex)
        name = st.text_input("Your first name (shown on the dashboard)", max_chars=30)
    with right:
        st.subheader("2 · What the models say")
        if photo is None:
            st.info("Waiting for a photo…")
        else:
            feats, thumb, cover = extract(photo)
            if feats is None:
                st.error("I couldn't find a banana: the photo looks almost colourless. Try a plain light background.")
            else:
                probs = models.predict(T, feats)
                avg = sum(probs.values()) / len(probs)
                verdict = "ripe" if avg >= 0.5 else "unripe"
                unusual, nov, reasons = T.familiar.check(feats)
                st.image(photo, width=220)
                if unusual:
                    why = f" ({', '.join(reasons)})" if reasons else ""
                    st.warning(f"⚠️ **This doesn't look like the bananas I was trained on**{why}. "
                               "The models below can only answer *ripe or unripe*, never *not a banana*, "
                               "so treat their verdict as a guess.")
                st.markdown(f"### Verdict: **{verdict.upper()}**  ({100*avg:.0f}% average chance of ripe)")
                st.dataframe(pd.DataFrame({"model": list(probs), "chance of ripe": [f"{100*p:.0f}%" for p in probs.values()],
                                           "test accuracy": [f"{100*T.accuracy[m]:.0f}%" for m in probs]}),
                             hide_index=True, use_container_width=True)
                with st.expander("The numbers the models saw"):
                    st.dataframe(pd.DataFrame({"feature": [FEATURE_LABELS[f] for f in FEATURES + [SHAPE]],
                                               "value": [feats[f] for f in FEATURES + [SHAPE]]}), hide_index=True)
                    st.caption(f"{100*cover:.0f}% of the photo looked like fruit. "
                               f"Unusualness score {nov:.1f} (99.5% of training photos score below {T.familiar.threshold:.1f}). "
                               "Roundness is not used for ripeness, only for this check.")
                key = f"saved_{hash(thumb)}"
                if not st.session_state.get(key):
                    if st.button("Add my banana to the community dashboard", type="primary"):
                        STORE.add(make_record(name, feats, probs, verdict, thumb, unusual=unusual))
                        st.session_state[key] = True
                        st.success("Added! Open the Community dashboard tab.")
                else:
                    st.success("Already added to the dashboard.")

# ---------------------------------------------------------------- 2. community dashboard
with tab2:
    up = STORE.all()
    a, b, c = st.columns(3)
    a.metric("Community uploads", len(up))
    b.metric("Training photos", len(load_seed()))
    c.metric("Storage", STORE.kind)
    st.caption(STORE_MSG)
    show(community.dashboard(load_seed(), up, T.road2d), "community")
    st.caption("Faint dots: the labelled photos the models learned from. Stars: community uploads (hover to see the photo). "
               "The line is a simple SVM drawn on just two of the six features.")

# ---------------------------------------------------------------- 3. rain board
with tab3:
    r, tab = rain_data()
    names = dict(zip(tab["name"], tab["SUBDIVISION"]))
    c1, c2 = st.columns([1, 2])
    pick = c1.selectbox("Region", list(names), index=list(names).index("Vidarbha"))
    yrs = c2.slider("Years", 1901, 2015, (1901, 2015))
    sub = names[pick]
    row_ = tab[tab.SUBDIVISION == sub].iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Average annual rain", f"{row_['mean']:,.0f} mm")
    m2.metric("Typical swing (SD)", f"{row_['sd']:,.0f} mm", help=f"That is {row_['cv']:.0f}% of the average.")
    m3.metric("Deficient years", f"{row_['deficient']} of {row_['years']}")
    show(rain.map_figure(tab, sub), "map")
    g1, g2 = st.columns(2)
    with g1:
        show(rain.trend_figure(r, sub, pick, yrs), "trend")
    with g2:
        show(rain.monthly_figure(r, sub, pick), "monthly")
    st.caption("Data: India Meteorological Department, subdivision rainfall 1901–2015 (data.gov.in). "
               "Deficient = at least 20% below the region's 1901–2015 average. Map positions are approximate centres.")

# ---------------------------------------------------------------- 4. model inspector
with tab4:
    st.subheader("A · Every classifier has a dial")
    st.write("A model gives a score. **You** choose the threshold that turns the score into a decision. "
             "Drag it and watch the mistakes trade places.")
    which = st.radio("Model to inspect", ["Weak model: sees only brightness", "Logistic regression: sees all six colour numbers"],
                     horizontal=True)
    if which.startswith("Weak"):
        st.caption(f"One number is not enough: this model gets {100*T.weak_accuracy:.0f}% right at threshold 0.5, "
                   "so the dial really matters.")
        show(inspector.threshold_demo(T.weak_scores, T.y_test, "brightness-only model"), "threshold_weak")
    else:
        st.caption(f"With all six numbers the task is easy ({100*T.accuracy['Logistic regression']:.0f}% right), "
                   "so almost any threshold works. Real problems are rarely this clean.")
        show(inspector.threshold_demo(T.test_scores["Logistic regression"], T.y_test), "threshold_full")
    st.subheader("B · A model that memorises")
    st.write("A more flexible curve always fits the training data better. Does it predict the hidden years better?")
    r, _ = rain_data()
    v = r[r.SUBDIVISION == "VIDARBHA"].dropna(subset=["ANNUAL"])
    lay, _ = inspector.overfit_demo(v["YEAR"], v["ANNUAL"])
    show(lay, "overfit")

# ---------------------------------------------------------------- 5. about
with tab5:
    st.markdown((DATA.parent / "ABOUT.md").read_text(encoding="utf-8"))

st.divider()
st.markdown(CREDIT)
