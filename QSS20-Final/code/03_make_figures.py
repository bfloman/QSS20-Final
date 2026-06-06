"""
03_make_figures.py -- Generate the four manuscript figures.

INPUT : data/processed/waiver_stateyears.csv  (for the live cross-validation figure)
OUTPUT: output/fig_withinparty.png      within-party margin slopes (Fig 1)
        output/fig_waiver_quartile.png  waiver prevalence by quartile (Fig 2)
        output/fig_crossval.png         in-sample vs leave-one-state-out F1 (Fig 3)
        output/fig_specforest.png       specification + placebo forest (Fig 4)

The cross-validation figure is computed live from the data; the within-party and
spec-forest figures are drawn from the validated estimates reported in 01_analysis.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score
from utils import PROCESSED_DIR, OUTPUT_DIR
import pandas as pd

SAVE = dict(dpi=160, bbox_inches="tight", pad_inches=0.03)


def fig_withinparty(path):
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    for lab, e, lo, hi, c, yi in [("Republican-controlled\n(n=21)", 75.5, 13, 138, "#b0392b", 1),
                                  ("Democratic-controlled\n(n=16)", 37.9, -75, 150, "#1f5fbf", 0)]:
        ax.plot([lo, hi], [yi, yi], color=c, lw=2); ax.plot(e, yi, "o", color=c, ms=6)
        ax.text(hi + 6, yi, f"{e:.0f}", va="center", fontsize=8, color=c)
    ax.axvline(0, ls="--", color="gray", lw=1)
    ax.set_yticks([1, 0]); ax.set_yticklabels(["Republican-controlled\n(n=21)", "Democratic-controlled\n(n=16)"], fontsize=8)
    ax.set_xlabel("Within-state margin slope on enrollment ratio (pp per unit)", fontsize=8.5)
    ax.tick_params(labelsize=8); ax.set_ylim(-0.6, 1.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, **SAVE); plt.close(fig)


def fig_waiver_quartile(path):
    fig, ax = plt.subplots(figsize=(6.4, 2.7)); x = np.arange(4)
    prev, lo, hi = [23.3, 11.1, 0, 0], [11, 4, 0, 0], [34, 22, 0, 0]
    ax.bar(x, prev, color=["#b0392b", "#d98a82", "#cdd6e0", "#9fb4d6"], edgecolor="black", lw=.6, width=.6)
    ax.errorbar(x, prev, yerr=[np.array(prev) - lo, np.array(hi) - np.array(prev)], fmt="none", ecolor="black", capsize=4, lw=.8)
    ax.plot([0], [8.0], marker="D", color="white", mec="black", ms=8, zorder=5)
    ax.annotate("Q1 without Indiana = 8.0%", xy=(0, 8.0), xytext=(0.5, 28), fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=.7))
    ax.set_xticks(x); ax.set_xticklabels(["Q1\n(most Rep.)", "Q2", "Q3", "Q4\n(most Dem.)"], fontsize=8)
    ax.set_ylabel("State-years w/ restrictive waiver (%)", fontsize=8.5); ax.tick_params(labelsize=8); ax.set_ylim(0, 38)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, **SAVE); plt.close(fig)


def fig_crossval(path):
    sm = pd.read_csv(os.path.join(PROCESSED_DIR, "waiver_stateyears.csv"))
    feats = ["margin", "dem_control", "gov_dem"]
    X, y, g = sm[feats].to_numpy(float), sm["restrictive_waiver"].to_numpy(), sm["state"].to_numpy()
    mk = lambda a, b: DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=0).fit(a, b)
    ins = f1_score(y, mk(X, y).predict(X))
    pr = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        pr[te] = mk(X[tr], y[tr]).predict(X[te])
    loso = f1_score(y, pr)
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    b = ax.bar(["In-sample\n(same states)", "Leave-one-state-out\n(held-out states)"], [ins, loso],
               color=["#4878a8", "#b0b7bf"], width=0.5, edgecolor="black", lw=.6)
    for bb, v in zip(b, [ins, loso]):
        ax.text(bb.get_x() + bb.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("F1, restrictive-waiver prediction", fontsize=8.5); ax.set_ylim(0, 1); ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, **SAVE); plt.close(fig)


def fig_specforest(path):
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    specs = ["(1) Full sample", "(2) Excl. COVID", "(3) Expansion states", "(4) Expansion, excl. COVID",
             "(A1) Non-expansion", "(A2) Non-exp., excl. COVID", "(B) Pre-expansion"]
    co = [80.85, 86.85, 93.95, 116.39, 24.71, 18.89, 3.82]
    se = [31.53, 35.79, 25.61, 27.22, 30.23, 31.38, 33.33]
    plac, yy = [0, 0, 0, 0, 1, 1, 1], np.arange(len(specs))[::-1]
    for yi, c, s, p in zip(yy, co, se, plac):
        col = "#9aa0a6" if p else "#2f5fa6"
        ax.plot([c - 1.96 * s, c + 1.96 * s], [yi, yi], color=col, lw=1.8); ax.plot(c, yi, "o", color=col, ms=5)
        ax.text(c + 1.96 * s + 4, yi, f"{c:.0f}", va="center", fontsize=7.5, color=col)
    ax.axvline(0, ls="--", color="gray", lw=1); ax.set_yticks(yy); ax.set_yticklabels(specs, fontsize=8)
    ax.set_xlabel("Coefficient on Dem. seat margin (pp per unit)", fontsize=8.5); ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, **SAVE); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig_withinparty(os.path.join(OUTPUT_DIR, "fig_withinparty.png"))
    fig_waiver_quartile(os.path.join(OUTPUT_DIR, "fig_waiver_quartile.png"))
    fig_crossval(os.path.join(OUTPUT_DIR, "fig_crossval.png"))
    fig_specforest(os.path.join(OUTPUT_DIR, "fig_specforest.png"))
    print(f"Wrote 4 figures to {OUTPUT_DIR}")
