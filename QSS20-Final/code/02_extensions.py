"""
02_extensions.py -- Restriction as a classification problem (paper centerpiece).

INPUT : data/processed/waiver_stateyears.csv  (from 00_build.py)
OUTPUT: console diagnostics that test whether majority SIZE predicts restriction
        beyond party DIRECTION:
          1. depth-3 decision tree    -> what does the model split on?
          2. leave-one-state-out CV   -> does the size pattern GENERALIZE?
          3. L1 (Lasso) logit         -> keep margin or party control?
          4. drop-Indiana refit       -> is the gradient one state?
          5. waiver-by-quartile       -> the descriptive gradient + its fragility
"""
import os
from math import sqrt
from statistics import NormalDist
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score, precision_score, recall_score
from utils import PROCESSED_DIR

FEATURES = ["margin", "dem_control", "gov_dem"]


def fit_tree(X, y):
    return DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=0).fit(X, y)


if __name__ == "__main__":
    sm = pd.read_csv(os.path.join(PROCESSED_DIR, "waiver_stateyears.csv"))
    X, y, groups = sm[FEATURES].to_numpy(float), sm["restrictive_waiver"].to_numpy(), sm["state"].to_numpy()
    print(f"Expansion state-years: {len(sm)} | restrictive-waiver years: {int(y.sum())}")

    # 1. interpretable tree
    tree = fit_tree(X, y)
    print("\n[1] Decision tree (depth 3):")
    print(export_text(tree, feature_names=FEATURES))
    print("    feature importances:", dict(zip(FEATURES, np.round(tree.feature_importances_, 3))))

    # 2. leave-one-state-out CV (grouped by state -> correct for panel data)
    preds = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        preds[te] = fit_tree(X[tr], y[tr]).predict(X[te])
    print("\n[2] Generalization (leave-one-state-out CV):")
    print(f"    in-sample F1 = {f1_score(y, tree.predict(X)):.2f}")
    print(f"    LOSO F1 = {f1_score(y, preds):.2f} | precision = {precision_score(y, preds, zero_division=0):.2f}"
          f" | recall = {recall_score(y, preds):.2f}")

    # 3. L1 (Lasso) logit: selects between direction and size; converges under separation
    l1 = LogisticRegression(penalty="l1", solver="liblinear", C=1.0).fit(X, y)
    print("\n[3] L1 logit coefficients:", dict(zip(FEATURES, np.round(l1.coef_[0], 3))),
          "-> margin shrunk to 0; party control retained")

    # 4. drop Indiana
    s2 = sm[sm["state"] != "Indiana"]
    t2 = fit_tree(s2[FEATURES].to_numpy(float), s2["restrictive_waiver"].to_numpy())
    print("\n[4] Refit WITHOUT Indiana -- importances:", dict(zip(FEATURES, np.round(t2.feature_importances_, 3))),
          "| predicted-waiver state-years:", int(t2.predict(s2[FEATURES].to_numpy(float)).sum()))

    # 5. waiver prevalence by margin quartile + Q1-vs-Q2 test + Indiana drop
    sm["q"] = pd.qcut(sm["margin"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    print("\n[5] Waiver prevalence by margin quartile (Q1 = most Republican):")
    print((sm.groupby("q", observed=True)["restrictive_waiver"].mean() * 100).round(1).to_string())
    q1, q2 = sm[sm.q == "Q1"]["restrictive_waiver"], sm[sm.q == "Q2"]["restrictive_waiver"]
    pp = (q1.sum() + q2.sum()) / (len(q1) + len(q2))
    z = (q1.mean() - q2.mean()) / sqrt(pp * (1 - pp) * (1 / len(q1) + 1 / len(q2)))
    print(f"    Q1={q1.mean()*100:.1f}% vs Q2={q2.mean()*100:.1f}%  z={z:.2f}, "
          f"p={2*(1-NormalDist().cdf(abs(z))):.3f} (independence assumed; optimistic)")
    print("    Q1 waiver-years by state:",
          dict(sm[(sm.q == 'Q1') & (sm.restrictive_waiver == 1)]['state'].value_counts()))
    q1b = sm[(sm.q == "Q1") & (sm.state != "Indiana")]["restrictive_waiver"]
    print(f"    drop Indiana -> Q1 = {q1b.mean()*100:.1f}% (now below Q2)")
