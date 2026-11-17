"""
models.py: three classic classifiers that decide "ripe" or "unripe".

FasalScope · © 2026 Dr. Nilesh P. Ingle · MIT licence

The models are retrained every time the app starts. The dataset is small,
so training takes under a second, and nothing has to be saved or version-matched.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import FEATURES

LABELS = ["unripe", "ripe"]          # 0, 1
PLOT_X, PLOT_Y = "pct_green", "pct_brown"   # the two features drawn on the dashboard


def make_models():
    return {
        # SVC gives a distance from the road, not a probability; calibration turns it into one.
        "SVM": make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(kernel="linear", C=1.0), ensemble=False)),
        "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    }


@dataclass
class Trained:
    models: dict          # name -> fitted pipeline (trained on ALL data)
    test_scores: dict     # name -> P(ripe) on the held-out test set
    y_test: np.ndarray
    accuracy: dict        # name -> test accuracy
    road2d: tuple         # (w1, w2, b) of a 2-feature linear SVM in original units
    n_train: int
    n_test: int


def train(df: pd.DataFrame, seed=42) -> Trained:
    X = df[FEATURES].to_numpy(float)
    y = (df["label"] == "ripe").astype(int).to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)

    scores, acc = {}, {}
    for name, m in make_models().items():
        m.fit(Xtr, ytr)
        scores[name] = m.predict_proba(Xte)[:, 1]
        acc[name] = accuracy_score(yte, m.predict(Xte))

    final = {name: m.fit(X, y) for name, m in make_models().items()}

    # A 2-feature SVM, only for drawing the "road" on the dashboard.
    X2 = df[[PLOT_X, PLOT_Y]].to_numpy(float)
    svm2 = SVC(kernel="linear", C=1.0).fit(X2, y)
    (w1, w2), b = svm2.coef_[0], svm2.intercept_[0]
    return Trained(final, scores, yte, acc, (float(w1), float(w2), float(b)), len(ytr), len(yte))


def predict(trained: Trained, feats: dict) -> dict:
    """name -> probability that the banana is ripe."""
    x = np.array([[feats[f] for f in FEATURES]], float)
    return {name: float(m.predict_proba(x)[0, 1]) for name, m in trained.models.items()}
