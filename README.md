# 🍌 FasalScope

**Photo → numbers → models → decision.** A small, complete data-science product: check whether a banana is ripe from a photo, see everyone's uploads on a live dashboard, explore 115 years of Indian rainfall on a map, and drag sliders to see how machine-learning models make (and miss) decisions.

Built by **Dr. Nilesh P. Ingle** for the course *1MTAIDS 33 Data Science* (Department of Applied Electronics, Sant Gadge Baba Amravati University, Amravati).
[LinkedIn](https://www.linkedin.com/in/nileshingle/)

| | |
|---|---|
| ▶ **Live app** | *link added after deployment* |
| ▶ **Look under the hood in Colab** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/learndataa/fasalscope/blob/main/notebooks/FasalScope_under_the_hood.ipynb) |
| ▶ **Make your own copy** | click **Use this template** (top right), then deploy it on Streamlit (steps below) |

---

## What's inside

| Tab | What it does | Course idea |
|---|---|---|
| **Check a banana** | Upload or snap a photo; three models say ripe or unripe | features, SVM, Naive Bayes, logistic regression |
| **Community dashboard** | Everyone's uploads as stars on a Bokeh chart, with the SVM "road" | MongoDB, linked Bokeh views, hover tooltips |
| **Rain Board** | IMD rainfall 1901–2015: map, 115-year trend, monsoon months | statistics, honest encodings, Bokeh maps |
| **Model Inspector** | Threshold slider with confusion matrix and ROC; overfitting slider | evaluation, train/test split, overfitting |

## How the pieces connect

```
Phone browser ──► Streamlit app (app.py) ◄──► MongoDB Atlas (uploads)
                        ▲
GitHub (this repo) ─────┘ auto-deploys on every change
        │
        └──► Google Colab notebook (explore data, retrain, experiment)
```

| File | Job |
|---|---|
| `app.py` | the page: tabs, photo upload, calls everything else |
| `features.py` | photo → six colour numbers (% green, % yellow, % brown, hue, saturation, brightness) |
| `models.py` | trains SVM, Naive Bayes and logistic regression when the app starts |
| `db.py` | saves and reads uploads (MongoDB if configured, otherwise temporary memory) |
| `community.py`, `rain.py`, `inspector.py` | the Bokeh charts for each tab |
| `build_dataset.py` | turns a folder of labelled photos into `data/banana_features.csv` |

## Run it on your computer

```bash
pip install -r requirements.txt
streamlit run app.py
```
It works without a database (uploads are kept until you stop the app).

## Deploy your own copy (free)

1. **Use this template** → create your repository.
2. Go to **share.streamlit.io**, sign in with GitHub, **Create app** → your repository, branch `main`, file `app.py`.
3. *(Optional, to keep uploads)* In **MongoDB Atlas** (free M0 cluster): create a database user, allow network access from `0.0.0.0/0`, copy the connection string. In Streamlit: **App settings → Secrets**, paste
   `MONGO_URI = "mongodb+srv://…"` (see `.streamlit/secrets.toml.example`). **Never commit the real string to GitHub.**

## Honest limits

The models only count colours, so lighting, background and banana variety all change the answer. FasalScope is a teaching project, not a certified agricultural tool. Uploads store six numbers, a 96-pixel thumbnail, a first name and the time; please photograph fruit only.

## Credits and licences

- **Code:** MIT licence © 2026 Dr. Nilesh P. Ingle. Reuse freely; keep the copyright notice.
- **Banana photos:** *Ripe Unripe Banana Dataset*, Mendeley Data, CC BY 4.0 (only the extracted colour numbers are stored here). See `data/README.md`.
- **Rainfall:** India Meteorological Department subdivision rainfall 1901–2015, published on data.gov.in.
- **Map tiles:** © OpenStreetMap contributors.
