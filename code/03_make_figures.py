"""
03_make_figures.py -- Generate the five manuscript figures (complete-data version).

INPUT : data/processed/waiver_stateyears.csv, data/raw/ncsl_lower.csv
OUTPUT: output/fig_data.png, fig_withinparty.png, fig_waiver_quartile.png,
        fig_crossval.png, fig_specforest.png
Cross-validation and the data figure are computed live; the FE point estimates
behind the within-party and spec-forest figures come from 01_analysis.py.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score
from utils import PROCESSED_DIR, OUTPUT_DIR, RAW_DIR

SAVE = dict(dpi=160, bbox_inches="tight", pad_inches=0.03)
ACC, BLUE, GREY = "#9a2b1f", "#2f5fa6", "#9aa0a6"


def fig_data(path):
    lo = pd.read_csv(os.path.join(RAW_DIR, "ncsl_lower.csv"))
    lo["margin"] = lo["dem_house_seats"] / lo["total_house_seats"] - 0.5
    yrs = [2015, 2017, 2019, 2021, 2023]
    piv = lo.pivot_table(index="state", columns="ncsl_year", values="margin").reindex(columns=yrs)
    flippers = [s for s, r in piv.iterrows() if len(set(np.sign(r.dropna().values))) > 1]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for st, row in piv.iterrows():
        if st in flippers:
            continue
        x = [y for y, v in zip(yrs, row.values) if not np.isnan(v)]
        ax.plot(x, row.dropna().values, color="#bcb19e", lw=.8, alpha=.5, zorder=1)
    for st in flippers:
        row = piv.loc[st]; x = [y for y, v in zip(yrs, row.values) if not np.isnan(v)]
        ax.plot(x, row.dropna().values, color=ACC, lw=2, zorder=3, marker="o", ms=3.5)
    ax.axhline(0, color="#1b1815", lw=1.1, zorder=2)
    ax.text(2014.75, .42, "Democratic majority", fontsize=8, color="#33545f", va="top", style="italic")
    ax.text(2014.75, -.42, "Republican majority", fontsize=8, color=ACC, va="bottom", style="italic")
    ax.annotate(f"{len(flippers)} chambers cross\nthe control line",
                xy=(2019, 0), xytext=(2019.4, .30), fontsize=8.5, color=ACC, fontweight="bold",
                ha="left", arrowprops=dict(arrowstyle="->", color=ACC, lw=1))
    ax.set_xticks(yrs); ax.set_yticks([-.4, -.2, 0, .2, .4]); ax.set_yticklabels(["−40", "−20", "0", "+20", "+40"])
    ax.set_ylabel("Dem. lower-chamber seat margin (pts)", fontsize=8.5)
    ax.tick_params(labelsize=8); ax.set_xlim(2014.6, 2023.5); ax.set_ylim(-.48, .48)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, **SAVE); plt.close(fig)


def fig_withinparty(path):
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    rows = [("Republican-controlled", 41.83, 30.08, ACC, 1),
            ("Democratic-controlled", -26.27, 49.55, BLUE, 0)]
    for lab, b, se, c, yi in rows:
        ax.plot([b - 1.96 * se, b + 1.96 * se], [yi, yi], color=c, lw=2)
        ax.plot(b, yi, "o", color=c, ms=6)
        ax.text(b + 1.96 * se + 6, yi, f"{b:.0f}", va="center", fontsize=8, color=c)
    ax.axvline(0, ls="--", color="gray", lw=1)
    ax.set_yticks([1, 0]); ax.set_yticklabels(["Republican-\ncontrolled", "Democratic-\ncontrolled"], fontsize=8)
    ax.set_xlabel("Within-state margin slope on enrollment ratio (pp per unit)", fontsize=8.5)
    ax.tick_params(labelsize=8); ax.set_ylim(-0.6, 1.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, **SAVE); plt.close(fig)


def fig_waiver_quartile(path):
    fig, ax = plt.subplots(figsize=(6.4, 2.7)); x = np.arange(4)
    prev, lo, hi = [14.7, 10.6, 3.8, 0], [7.9, 4.7, 0, 0], [21.5, 16.5, 7.7, 0]
    ax.bar(x, prev, color=[ACC, "#d98a82", "#cdd6e0", "#9fb4d6"], edgecolor="black", lw=.6, width=.6)
    ax.errorbar(x, prev, yerr=[np.array(prev) - lo, np.array(hi) - np.array(prev)], fmt="none", ecolor="black", capsize=4, lw=.8)
    ax.plot([0], [4.8], marker="D", color="white", mec="black", ms=8, zorder=5)
    ax.annotate("Q1 without Indiana = 4.8%", xy=(0, 4.8), xytext=(0.5, 20), fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=.7))
    ax.set_xticks(x); ax.set_xticklabels(["Q1\n(most Rep.)", "Q2", "Q3", "Q4\n(most Dem.)"], fontsize=8)
    ax.set_ylabel("State-years w/ restrictive waiver (%)", fontsize=8.5); ax.tick_params(labelsize=8); ax.set_ylim(0, 24)
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
    return ins, loso


def fig_specforest(path):
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    specs = ["(1) Full sample", "(2) Excl. COVID", "(3) Expansion states",
             "(4) Expansion, excl. COVID", "(A) Non-expansion [placebo]", "(B) Pre-expansion [placebo]"]
    co = [30.74, 38.93, 33.08, 42.21, 29.63, 40.47]
    se = [19.65, 22.28, 19.59, 22.89, 33.40, 16.62]
    plac, yy = [0, 0, 0, 0, 1, 1], np.arange(len(specs))[::-1]
    for yi, c, s, p in zip(yy, co, se, plac):
        col = GREY if p else BLUE
        ax.plot([c - 1.96 * s, c + 1.96 * s], [yi, yi], color=col, lw=1.8); ax.plot(c, yi, "o", color=col, ms=5)
        ax.text(c + 1.96 * s + 3, yi, f"{c:.0f}", va="center", fontsize=7.5, color=col)
    ax.axvline(0, ls="--", color="gray", lw=1); ax.set_yticks(yy); ax.set_yticklabels(specs, fontsize=8)
    ax.set_xlabel("Coefficient on Dem. seat margin (pp per unit)", fontsize=8.5); ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, **SAVE); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig_data(os.path.join(OUTPUT_DIR, "fig_data.png"))
    fig_withinparty(os.path.join(OUTPUT_DIR, "fig_withinparty.png"))
    fig_waiver_quartile(os.path.join(OUTPUT_DIR, "fig_waiver_quartile.png"))
    ins, loso = fig_crossval(os.path.join(OUTPUT_DIR, "fig_crossval.png"))
    fig_specforest(os.path.join(OUTPUT_DIR, "fig_specforest.png"))
    print(f"Wrote 5 figures. Cross-val F1: in={ins:.2f} loso={loso:.2f}")
