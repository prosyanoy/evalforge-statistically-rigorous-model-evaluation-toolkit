"""Bundled demo datasets (shipped inside scikit-learn, so no network access) with
a baseline and a candidate model each, trained on a stratified split."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATASETS = ("breast_cancer", "wine", "digits")


@dataclass
class DemoSplit:
    name: str
    description: str
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    classes: list
    proba_candidate: object
    proba_baseline: object
    candidate_name: str
    baseline_name: str


def _load(name: str):
    if name == "breast_cancer":
        d = load_breast_cancer(as_frame=True)
        desc = "Wisconsin breast cancer diagnostics (binary, 30 numeric features)."
        cand = ("StandardScaler + LogisticRegression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)))
        base = ("DecisionTree(max_depth=3)", DecisionTreeClassifier(max_depth=3, random_state=0))
    elif name == "wine":
        d = load_wine(as_frame=True)
        desc = "UCI wine recognition (3 cultivars, 13 numeric features, only 178 rows)."
        cand = ("StandardScaler + LogisticRegression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)))
        base = ("GaussianNB", GaussianNB())
    elif name == "digits":
        d = load_digits(as_frame=True)
        desc = "8x8 handwritten digits (10 classes, 64 pixel features)."
        cand = ("RandomForest(20, max_depth=3)", RandomForestClassifier(n_estimators=20, max_depth=3, random_state=0))
        base = ("GaussianNB", GaussianNB())
    else:
        raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")
    return d.data, d.target, [str(c) for c in d.target_names] if hasattr(d, "target_names") else None, desc, cand, base


def load_demo(name: str, test_size: float = 0.4, seed: int = 0) -> DemoSplit:
    X, y, names, desc, (cname, cand), (bname, base) = _load(name)
    if names is None:
        names = [str(c) for c in sorted(y.unique())]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    cand.fit(Xtr, ytr)
    base.fit(Xtr, ytr)
    return DemoSplit(name, desc, Xtr.reset_index(drop=True), Xte.reset_index(drop=True), ytr.reset_index(drop=True),
                     yte.reset_index(drop=True), list(range(len(names))), cand.predict_proba(Xte),
                     base.predict_proba(Xte), cname, bname)
