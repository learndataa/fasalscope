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

from features import FEATURES, FEATURE_LABELS, SHAPE

LABELS = ["unripe", "ripe"]          # 0, 1
PLOT_X, PLOT_Y = "pct_green", "pct_yellow"   # the two features drawn on the dashboard
WEAK_FEATURE = "mean_val"                     # the single feature used by the weak demo model


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
    weak_scores: np.ndarray = None   # a deliberately weak model (brightness only), for the threshold demo
    weak_accuracy: float = None
    familiar: "Familiarity" = None   # "does this look like my training photos?"


def train(df: pd.DataFrame, seed=42) -> Trained:
    X = df[FEATURES].to_numpy(float)
    y = (df["label"] == "ripe").astype(int).to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)

    scores, acc = {}, {}
    for name, m in make_models().items():
        m.fit(Xtr, ytr)
        scores[name] = m.predict_proba(Xte)[:, 1]
        acc[name] = accuracy_score(yte, m.predict(Xte))

    # A deliberately weak model that sees only one number (average brightness).
    # Its scores overlap, which makes the threshold trade-off easy to see.
    j = FEATURES.index(WEAK_FEATURE)
    weak = make_pipeline(StandardScaler(), LogisticRegression()).fit(Xtr[:, [j]], ytr)
    weak_scores = weak.predict_proba(Xte[:, [j]])[:, 1]
    weak_acc = accuracy_score(yte, weak.predict(Xte[:, [j]]))

    final = {name: m.fit(X, y) for name, m in make_models().items()}

    # A 2-feature SVM, only for drawing the "road" on the dashboard.
    X2 = df[[PLOT_X, PLOT_Y]].to_numpy(float)
    svm2 = SVC(kernel="linear", C=1.0).fit(X2, y)
    (w1, w2), b = svm2.coef_[0], svm2.intercept_[0]
    return Trained(final, scores, yte, acc, (float(w1), float(w2), float(b)), len(ytr), len(yte),
                   weak_scores, weak_acc, Familiarity(df))


def predict(trained: Trained, feats: dict) -> dict:
    """name -> probability that the banana is ripe."""
    x = np.array([[feats[f] for f in FEATURES]], float)
    return {name: float(m.predict_proba(x)[0, 1]) for name, m in trained.models.items()}


# ------------------------------------------------------------------ "is this even a banana?"
NOVELTY_COLS = ["pct_green", "pct_yellow", "mean_hue", "mean_sat", "mean_val", SHAPE]   # (% brown is left out:
                                                                                     # green + yellow + brown = 100)
REASONS = {"pct_green": ("very green", "hardly any green"), "pct_yellow": ("very yellow", "hardly any yellow"),
           "mean_hue": ("unusual colour", "unusual colour"), "mean_sat": ("unusually vivid colour", "unusually dull colour"),
           "mean_val": ("unusually bright", "unusually dark"), SHAPE: ("unusually long and thin", "too round for a banana")}


class Familiarity:
    """
    How far is a new photo from the training photos? We measure the distance to the nearest class
    (ripe or unripe) in a way that allows for how much each number usually varies and for numbers
    that move together (the Mahalanobis distance). The cut-off is set so that 99.5% of the
    training photos count as familiar.
    """

    def __init__(self, df, quantile=99.5):
        from sklearn.covariance import LedoitWolf
        self.fits, self.stats = {}, {}
        for lab, g in df.groupby("label"):
            X = g[NOVELTY_COLS].to_numpy(float)
            self.fits[lab] = LedoitWolf().fit(X)
            self.stats[lab] = (X.mean(0), X.std(0) + 1e-9)
        self.threshold = float(np.percentile(self.score_many(df[NOVELTY_COLS].to_numpy(float)), quantile))

    def score_many(self, X):
        return np.sqrt(np.min([f.mahalanobis(X) for f in self.fits.values()], axis=0))

    def check(self, feats):
        """Return (unusual: bool, score: float, reasons: list of short phrases)."""
        x = np.array([[feats[c] for c in NOVELTY_COLS]], float)
        score = float(self.score_many(x)[0])
        nearest = min(self.fits, key=lambda k: self.fits[k].mahalanobis(x)[0])
        mu, sd = self.stats[nearest]
        z = (x[0] - mu) / sd
        order = np.argsort(-np.abs(z))
        reasons = [REASONS[NOVELTY_COLS[i]][0 if z[i] > 0 else 1] for i in order[:2] if abs(z[i]) > 2]
        return score > self.threshold, score, reasons
