"""
QSS 20 Final Project — Extension Analyses
Five additional analyses extending the main pipeline.

Author: Ben Floman | Dartmouth College | QSS 20 | May 2026

Additional data files required in data/:
  ncsl_upper.csv          ← NCSL upper chamber (Senate) partisan composition
  governor_party.csv      ← Governor party by state and NCSL session year
  restrictive_waivers.csv ← State-years with active restrictive Medicaid waivers

Usage:
    python extensions.py                     # run all five extensions
    python extensions.py --only event_study  # run a single extension
    python extensions.py --only waiver rd    # run two extensions
    python extensions.py --rebuild-panel     # force panel rebuild first

Available --only values:
    event_study  supermajority_rd  waiver  bicameral  unified_gov
"""

import argparse
import logging
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2

warnings.filterwarnings(
    "ignore", message=".*Optimization terminated successfully.*", category=UserWarning
)
warnings.filterwarnings(
    "ignore", message=".*kurtosistest.*", category=UserWarning
)

# ── Import base pipeline ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis import (
    load_ncsl_lower, load_expansion_dates,
    load_or_build_panel, _save_fig,
    DATA_DIR, OUTPUT_DIR, data_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# EXTENSION DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_ncsl_upper(path: str = None) -> pd.DataFrame:
    """
    NCSL upper chamber (Senate) partisan composition at 5 election-year snapshots.
    Columns: state, ncsl_year, dem_senate_seats, total_senate_seats
    Returns: adds dem_share_senate, margin_senate
    """
    path = path or data_path("ncsl_upper.csv")
    df = pd.read_csv(path)
    df["dem_share_senate"] = df["dem_senate_seats"] / df["total_senate_seats"]
    df["margin_senate"]    = df["dem_share_senate"] - 0.5
    return df


def load_governor_party(path: str = None) -> pd.DataFrame:
    """
    Governor party at each NCSL session snapshot.
    Columns: state, ncsl_year, gov_dem  (1=Democrat, 0=Republican)
    """
    path = path or data_path("governor_party.csv")
    return pd.read_csv(path)


def load_waiver_data(path: str = None,
                     exp_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    State-years with an active restrictive Medicaid waiver.
    Source: KFF Medicaid Waiver Tracker.
    Columns in source: state, year  (sparse — only rows with active waiver)

    exp_df is accepted as an argument to avoid re-reading expansion_dates.csv
    when the caller already has it in memory. Falls back to loading it if None.

    Returns: full state-year panel with restrictive_waiver (0/1).
    """
    path = path or data_path("restrictive_waivers.csv")
    active = pd.read_csv(path)

    if exp_df is None:
        exp_df = load_expansion_dates()

    all_states = exp_df["state"].tolist()
    full = pd.DataFrame(
        {"state": s, "year": y}
        for s in all_states
        for y in range(2014, 2027)
    )

    full = full.merge(
        active[["state", "year"]].assign(restrictive_waiver=1),
        on=["state", "year"], how="left"
    )
    full["restrictive_waiver"] = full["restrictive_waiver"].fillna(0).astype(int)
    return full


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 1: EVENT STUDY AROUND PARTISAN CONTROL FLIPS
# ─────────────────────────────────────────────────────────────────────────────

def run_event_study(panel: pd.DataFrame, ncsl_df: pd.DataFrame,
                    outdir: str = OUTPUT_DIR) -> None:
    """
    Event study centered on elections that changed lower chamber control.

    Demeaning uses the PRE-PERIOD mean (t ≤ -1) within each state,
    not the full ±4-year window. Using the full window would pull the
    pre-period estimates downward (toward zero) when post-period levels
    are substantially higher, creating the appearance of a more dramatic
    pre-trend reversal than actually exists.

    Also runs a formal joint pre-trend test: tests H₀ that all t ≤ -1
    coefficients are jointly zero using a χ² Wald test. This is reported
    in-console as support for (or against) the parallel-trends assumption.
    """
    ncsl = ncsl_df.copy().sort_values(["state", "ncsl_year"])
    ncsl["prev_control"] = ncsl.groupby("state")["dem_control"].shift(1)
    ncsl["flipped"] = (
        (ncsl["dem_control"] != ncsl["prev_control"]) &
        ncsl["prev_control"].notna()
    )
    flip_events = ncsl[ncsl["flipped"]][["state", "ncsl_year", "dem_control"]].copy()
    flip_events["flip_to"] = flip_events["dem_control"].map({1: "→ Dem", 0: "→ Rep"})

    log.info("[EVENT STUDY] %d partisan control flip events:", len(flip_events))
    print(flip_events.to_string(index=False))

    # Annual panel
    annual = (
        panel.copy()
        .assign(year=lambda d: d["date"].dt.year)
        .groupby(["state", "year"])
        .agg(takeup=("enroll_rate", "mean"),
             margin=("margin", "first"),
             dem_control=("dem_control", "first"))
        .reset_index()
    )

    event_rows = []
    for _, ev in flip_events.iterrows():
        sub = annual[annual["state"] == ev["state"]].copy()
        sub["event_time"] = sub["year"] - ev["ncsl_year"]
        sub["flip_to"]    = ev["flip_to"]
        event_rows.append(sub)

    ev_df = (
        pd.concat(event_rows, ignore_index=True)
        .query("event_time >= -4 and event_time <= 4")
    )

    # ── Demean using PRE-PERIOD (t ≤ -1) mean only ────────────────────────────
    pre_means = (
        ev_df[ev_df["event_time"] <= -1]
        .groupby("state")["takeup"]
        .mean()
        .rename("pre_mean")
    )
    ev_df = ev_df.merge(pre_means, on="state", how="left")
    # Fallback for states with no pre-period observations: use full mean
    full_means = ev_df.groupby("state")["takeup"].mean().rename("full_mean")
    ev_df = ev_df.merge(full_means, on="state", how="left")
    ev_df["demeaning_base"] = ev_df["pre_mean"].fillna(ev_df["full_mean"])
    ev_df["takeup_dm"] = ev_df["takeup"] - ev_df["demeaning_base"]

    # ── Parallel-trends test: Wald test of H₀: β_{t≤-1} = 0 jointly ─────────
    def pre_trend_test(sub: pd.DataFrame, direction: str) -> None:
        """
        Regress demeaned take-up on event-time dummies (pre-period only)
        and perform a joint Wald test. Prints the χ² statistic and p-value.
        """
        pre = sub[sub["event_time"] <= -1].copy()
        if len(pre) < 3 or pre["event_time"].nunique() < 2:
            log.info("  %s: insufficient pre-period data for parallel-trends test.",
                     direction)
            return
        pre["et_dummy"] = pre["event_time"].astype(str)
        try:
            m = smf.ols("takeup_dm ~ C(et_dummy)", data=pre).fit()
            # Joint test: all event-time dummies = 0
            restrictions = [f"C(et_dummy)[T.{t}] = 0"
                            for t in sorted(pre["event_time"].unique())[1:]]
            if not restrictions:
                return
            ftest = m.f_test(restrictions)
            chi2_stat = ftest.statistic * len(restrictions)
            pval      = 1 - chi2.cdf(chi2_stat, df=len(restrictions))
            log.info("  %s pre-trend test: χ²(%d) = %.2f, p = %.3f — %s",
                     direction, len(restrictions), chi2_stat, pval,
                     "✓ consistent with parallel trends" if pval > 0.1
                     else "⚠ pre-trends detected")
        except Exception as exc:
            log.warning("  Pre-trend test failed for %s: %s", direction, exc)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, direction, color, title in [
        (axes[0], "→ Dem", "#1f77b4", "Flips to Democratic Control"),
        (axes[1], "→ Rep", "#d62728",  "Flips to Republican Control"),
    ]:
        sub = ev_df[ev_df["flip_to"] == direction]
        if sub.empty:
            ax.set_title(f"{title}\n(no events)")
            continue

        pre_trend_test(sub, direction)

        grp = (
            sub.groupby("event_time")["takeup_dm"]
            .agg(["mean", "sem"])
            .reset_index()
        )
        n_states = sub["state"].nunique()
        n_events = sub[sub["event_time"] == 0]["state"].nunique()

        ax.axvline(0, color="black", linestyle="--", linewidth=1.0,
                   label="Election (t=0)")
        ax.fill_between(
            grp["event_time"],
            grp["mean"] - 1.96 * grp["sem"],
            grp["mean"] + 1.96 * grp["sem"],
            alpha=0.20, color=color
        )
        ax.plot(grp["event_time"], grp["mean"],
                color=color, marker="o", linewidth=2.0, markersize=5)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_title(f"{title}\n(n={n_states} states, {n_events} events)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Years Relative to Election (t=0)", fontsize=10)
        ax.set_ylabel("Take-Up Rate (demeaned, pp)", fontsize=10)
        ax.legend(fontsize=9)
        ax.set_xticks(range(-4, 5))
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Event Study: Medicaid Take-Up Rate Around Partisan Control Flips\n"
        "Demeaned on pre-period state mean (t ≤ −1); 95% confidence intervals",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    _save_fig(fig, os.path.join(outdir, "fig4_event_study.png"))


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 2: SUPERMAJORITY THRESHOLD RD
# ─────────────────────────────────────────────────────────────────────────────

def run_supermajority_rd(panel: pd.DataFrame, outdir: str = OUTPUT_DIR) -> None:
    """
    Regression discontinuity at the Democratic supermajority threshold
    (dem_share = 0.60 → margin = 0.10).

    Uses statsmodels OLS with an indicator × running-variable interaction
    to obtain a formal discontinuity estimate with a standard error and
    confidence interval, rather than just the difference in scipy linregress
    intercepts (which has no associated uncertainty estimate).

    Model: takeup = α + β₁·above + β₂·rv + β₃·(above × rv) + ε
    The discontinuity estimate is β₁ (jump at the threshold).
    """
    THRESHOLD = 0.10    # margin = 0.10 → dem_share = 0.60
    BANDWIDTH = 0.20

    grp = (
        panel
        .groupby(["state", "ncsl_year"])
        .agg(takeup=("enroll_rate", "mean"), margin=("margin", "first"))
        .reset_index()
    )
    grp["rv"]    = grp["margin"] - THRESHOLD
    grp["above"] = (grp["rv"] >= 0).astype(int)
    rd = grp[grp["rv"].abs() <= BANDWIDTH].copy()

    n_above = rd["above"].sum()
    n_below = len(rd) - n_above
    log.info("[SUPERMAJORITY RD] threshold=%.2f  bandwidth=±%.2f  "
             "N=%d (%d above, %d below)",
             0.5 + THRESHOLD, BANDWIDTH, len(rd), n_above, n_below)

    # ── Formal RD estimate ────────────────────────────────────────────────────
    try:
        m = smf.ols("takeup ~ above + rv + above:rv", data=rd).fit()
        gap    = m.params["above"]
        gap_se = m.bse["above"]
        gap_p  = m.pvalues["above"]
        log.info("  RD discontinuity: %.2f pp  (SE=%.2f, p=%.3f, 95%% CI [%.2f, %.2f])",
                 gap, gap_se, gap_p,
                 gap - 1.96 * gap_se, gap + 1.96 * gap_se)
    except Exception as exc:
        log.warning("  RD model failed: %s", exc)
        gap = None

    # ── Figure ────────────────────────────────────────────────────────────────
    below = rd[rd["above"] == 0]
    above = rd[rd["above"] == 1]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(below["rv"], below["takeup"], color="#d62728",
               alpha=0.45, s=25, label="Below supermajority")
    ax.scatter(above["rv"], above["takeup"], color="#1f77b4",
               alpha=0.45, s=25, label="At/above supermajority")

    for sub, color in [(below, "#d62728"), (above, "#1f77b4")]:
        if len(sub) >= 3:
            # Use the interacted OLS fit for the plot line (not linregress)
            sub_m = smf.ols("takeup ~ rv", data=sub).fit()
            x_fit = np.linspace(sub["rv"].min(), sub["rv"].max(), 100)
            ax.plot(x_fit,
                    sub_m.params["Intercept"] + sub_m.params["rv"] * x_fit,
                    color=color, linewidth=2.0)

    ax.axvline(0, color="black", linestyle="--", linewidth=1.2,
               label="Supermajority threshold (60%)")
    if gap is not None:
        ax.annotate(
            f"Gap = {gap:+.1f} pp\n(SE={gap_se:.1f}, p={gap_p:.2f})",
            xy=(0, rd["takeup"].median()),
            xytext=(0.04, rd["takeup"].median() - 15),
            fontsize=9, color="black",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        )
    ax.set_xlabel("Dem. Seat Margin − Supermajority Threshold (0.60)", fontsize=11)
    ax.set_ylabel("Mean Medicaid Take-Up Rate (%)", fontsize=11)
    ax.set_title("Regression Discontinuity at the Supermajority Threshold\n"
                 "Session-Level Averages, ±0.20 Bandwidth",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, os.path.join(outdir, "fig5_supermajority_rd.png"))


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 3: RESTRICTIVE WAIVERS AS ALTERNATIVE OUTCOME
# ─────────────────────────────────────────────────────────────────────────────

def run_waiver_analysis(panel: pd.DataFrame, waiver_df: pd.DataFrame,
                        outdir: str = OUTPUT_DIR) -> None:
    """
    Does Democratic seat margin predict the absence of restrictive Medicaid waivers?
    Logit (year FE) + LPM (state + year FE) on expansion states only.
    """
    annual = (
        panel[panel["post_expansion"] == 1].copy()
        .assign(year=lambda d: d["date"].dt.year)
        .groupby(["state", "year"])
        .agg(margin=("margin", "first"),
             dem_control=("dem_control", "first"),
             mood=("mood", "first"),
             takeup=("enroll_rate", "mean"))
        .reset_index()
    )

    annual = annual.merge(waiver_df[["state", "year", "restrictive_waiver"]],
                          on=["state", "year"], how="left")
    annual["restrictive_waiver"] = annual["restrictive_waiver"].fillna(0)
    annual["state_f"] = annual["state"].astype("category")
    annual["year_f"]  = annual["year"].astype("category")

    n_waiver = int(annual["restrictive_waiver"].sum())
    log.info("[WAIVER ANALYSIS] Expansion state-years: %d  |  "
             "With restrictive waiver: %d (%.1f%%)",
             len(annual), n_waiver, 100 * n_waiver / len(annual))

    for spec, formula, fit_kwargs in [
        ("Logit (year FE)",
         "restrictive_waiver ~ margin + dem_control + mood + C(year_f)",
         {"disp": 0}),
        ("LPM (state+year FE)",
         "restrictive_waiver ~ margin + dem_control + mood "
         "+ C(state_f) + C(year_f)",
         {"cov_type": "cluster",
          "cov_kwds": {"groups": annual["state"]}}),
    ]:
        fn = smf.logit if "Logit" in spec else smf.ols
        try:
            m = fn(formula, data=annual).fit(**fit_kwargs)
            print(f"\n  {spec}:")
            print(f"  {'Variable':<18} {'Coef':>9} {'SE':>8} {'p':>8} "
                  f"{'95% CI Lo':>11} {'95% CI Hi':>11}")
            for v in ["margin", "dem_control", "mood"]:
                if v in m.params.index:
                    lo = m.params[v] - 1.96 * m.bse[v]
                    hi = m.params[v] + 1.96 * m.bse[v]
                    print(f"  {v:<18} {m.params[v]:>9.3f} {m.bse[v]:>8.3f} "
                          f"{m.pvalues[v]:>8.3f} {lo:>11.3f} {hi:>11.3f}")
            stat = getattr(m, "prsquared", None) or getattr(m, "rsquared", None)
            print(f"  N={int(m.nobs)}, R²/PseudoR²={stat:.3f}")
        except Exception as exc:
            log.error("  %s failed: %s", spec, exc)

    # ── Figure: waiver rate by margin quartile ─────────────────────────────────
    annual["margin_q"] = pd.qcut(
        annual["margin"], 4,
        labels=["Q1\n(Most Rep.)", "Q2", "Q3", "Q4\n(Most Dem.)"]
    )
    qgrp = (annual.groupby("margin_q")["restrictive_waiver"]
            .agg(["mean", "sem"])
            .reset_index())

    fig, ax = plt.subplots(figsize=(8, 5))
    colors_q = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
    ax.bar(qgrp["margin_q"].astype(str), qgrp["mean"] * 100,
           color=colors_q, alpha=0.8, width=0.5)
    ax.errorbar(range(4), qgrp["mean"] * 100,
                yerr=qgrp["sem"] * 100 * 1.96, fmt="none",
                color="black", capsize=4)
    ax.set_xlabel("Democratic Seat Margin Quartile", fontsize=11)
    ax.set_ylabel("Share of State-Years with Restrictive Waiver (%)", fontsize=11)
    ax.set_title("Restrictive Medicaid Waivers by Legislative Margin Quartile\n"
                 "Expansion States Only", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, os.path.join(outdir, "fig6_waiver_by_margin.png"))


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 4: BICAMERAL MARGIN
# ─────────────────────────────────────────────────────────────────────────────

def run_bicameral_analysis(panel: pd.DataFrame, upper_df: pd.DataFrame,
                            outdir: str = OUTPUT_DIR) -> None:
    """
    Compare lower-only, upper-only, binding (min), and average chamber margin.
    Tsebelis predicts the binding (minimum) chamber is the operative constraint.
    """
    p2 = panel.merge(upper_df[["state", "ncsl_year", "margin_senate"]],
                     on=["state", "ncsl_year"], how="left")
    p2["margin_binding"] = p2[["margin", "margin_senate"]].min(axis=1)
    p2["margin_avg"]     = p2[["margin", "margin_senate"]].mean(axis=1)
    p2 = p2.dropna(subset=["margin_senate", "enroll_rate", "mood"])
    p2["state_f"] = p2["state"].astype("category")
    p2["year_f"]  = p2["date"].dt.year.astype("category")

    specs = [
        ("Lower chamber only",    "margin"),
        ("Senate only",           "margin_senate"),
        ("Binding (min) chamber", "margin_binding"),
        ("Average of chambers",   "margin_avg"),
    ]

    results = {}
    log.info("[BICAMERAL ANALYSIS]")
    for label, var in specs:
        try:
            m = smf.ols(
                f"enroll_rate ~ {var} + dem_control + post_expansion + mood"
                " + C(state_f) + C(year_f)",
                data=p2
            ).fit(cov_type="cluster", cov_kwds={"groups": p2["state"]})
            c, s, pv = m.params[var], m.bse[var], m.pvalues[var]
            ci_lo, ci_hi = c - 1.96 * s, c + 1.96 * s
            log.info("  %-30s coef=%8.3f  se=%7.3f  p=%6.3f  "
                     "95%%CI [%.2f, %.2f]  N=%d",
                     label, c, s, pv, ci_lo, ci_hi, int(m.nobs))
            results[label] = (c, s, pv)
        except Exception as exc:
            log.error("  %s: FAILED — %s", label, exc)

    # ── Coefficient plot ──────────────────────────────────────────────────────
    labels_p = list(results.keys())
    coefs    = [results[l][0] for l in labels_p]
    ses      = [results[l][1] for l in labels_p]
    pvals    = [results[l][2] for l in labels_p]
    bar_cols = ["#1f77b4" if p < 0.1 else "#aec7e8" for p in pvals]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(labels_p)), coefs,
            xerr=[1.96 * s for s in ses],
            color=bar_cols, alpha=0.8, capsize=4, height=0.5)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(range(len(labels_p)))
    ax.set_yticklabels(labels_p, fontsize=10)
    ax.set_xlabel("Coefficient on Margin Variable (pp per unit)", fontsize=11)
    ax.set_title("Bicameral Margin Specifications\n"
                 "Two-Way FE, Clustered SE, Bars = 95% CI",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, os.path.join(outdir, "fig7_bicameral.png"))


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 5: UNIFIED GOVERNMENT INTERACTION
# ─────────────────────────────────────────────────────────────────────────────

def run_unified_gov_analysis(panel: pd.DataFrame, gov_df: pd.DataFrame,
                              outdir: str = OUTPUT_DIR) -> None:
    """
    Does the margin effect depend on whether the governor shares the
    legislative majority's party? Interaction: margin × unified_gov.
    """
    from scipy.stats import linregress as lr

    p2 = panel.merge(gov_df[["state", "ncsl_year", "gov_dem"]],
                     on=["state", "ncsl_year"], how="left")
    p2 = p2.dropna(subset=["gov_dem", "enroll_rate", "mood"])

    p2["unified_dem"] = ((p2["dem_control"] == 1) & (p2["gov_dem"] == 1)).astype(int)
    p2["unified_rep"] = ((p2["dem_control"] == 0) & (p2["gov_dem"] == 0)).astype(int)
    p2["unified"]     = (p2["unified_dem"] | p2["unified_rep"]).astype(int)

    # Vectorized government type label (replaces row-by-row apply)
    p2["gov_type"] = np.select(
        [p2["unified_dem"] == 1, p2["unified_rep"] == 1],
        ["Unified Dem", "Unified Rep"],
        default="Divided"
    )

    p2["state_f"] = p2["state"].astype("category")
    p2["year_f"]  = p2["date"].dt.year.astype("category")

    log.info("[UNIFIED GOVERNMENT INTERACTION]")
    log.info("  Unified Dem: %d state-months", p2["unified_dem"].sum())
    log.info("  Unified Rep: %d state-months", p2["unified_rep"].sum())
    log.info("  Divided:     %d state-months", (~p2["unified"].astype(bool)).sum())

    # ── Interaction model ─────────────────────────────────────────────────────
    try:
        m = smf.ols(
            "enroll_rate ~ margin * unified + dem_control + post_expansion"
            " + mood + C(state_f) + C(year_f)",
            data=p2
        ).fit(cov_type="cluster", cov_kwds={"groups": p2["state"]})
        print("\n  Interaction model: margin × unified government")
        print(f"  {'Variable':<25} {'Coef':>9} {'SE':>8} {'p':>8}")
        for v in ["margin", "unified", "margin:unified",
                  "dem_control", "post_expansion"]:
            if v in m.params.index:
                print(f"  {v:<25} {m.params[v]:>9.3f} {m.bse[v]:>8.3f} "
                      f"{m.pvalues[v]:>8.3f}")
    except Exception as exc:
        log.error("  Interaction model failed: %s", exc)

    # ── Subgroup regressions ──────────────────────────────────────────────────
    print("\n  Subgroup regressions:")
    for label, sub in [("Unified gov.", p2[p2["unified"] == 1]),
                        ("Divided gov.", p2[p2["unified"] == 0])]:
        try:
            m_s = smf.ols(
                "enroll_rate ~ margin + dem_control + post_expansion"
                " + mood + C(state_f) + C(year_f)",
                data=sub
            ).fit(cov_type="cluster", cov_kwds={"groups": sub["state"]})
            log.info("  %-20s margin coef=%8.3f  se=%7.3f  p=%6.3f  N=%d",
                     label, m_s.params["margin"], m_s.bse["margin"],
                     m_s.pvalues["margin"], int(m_s.nobs))
        except Exception as exc:
            log.error("  %s: FAILED — %s", label, exc)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    for ax, gtype, col in zip(axes,
                               ["Unified Dem", "Divided", "Unified Rep"],
                               ["#1f77b4", "#7f7f7f", "#d62728"]):
        sub = p2[p2["gov_type"] == gtype]
        if sub.empty:
            ax.set_visible(False)
            continue
        ax.scatter(sub["margin"], sub["enroll_rate"],
                   color=col, alpha=0.08, s=5)
        slope, intercept, _, pv, _ = lr(sub["margin"], sub["enroll_rate"])
        x_fit = np.linspace(sub["margin"].min(), sub["margin"].max(), 100)
        ax.plot(x_fit, intercept + slope * x_fit, color=col, linewidth=2.0)
        ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(f"{gtype}\nslope={slope:.1f}, p={pv:.2f}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Dem. Seat Margin", fontsize=9)

    axes[0].set_ylabel("Medicaid Take-Up Rate (%)", fontsize=10)
    fig.suptitle("Legislative Margin vs. Take-Up Rate by Government Type",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, os.path.join(outdir, "fig8_unified_gov.png"))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

_ALL_ANALYSES = {
    "event_study":      "run_event_study",
    "supermajority_rd": "run_supermajority_rd",
    "waiver":           "run_waiver_analysis",
    "bicameral":        "run_bicameral_analysis",
    "unified_gov":      "run_unified_gov_analysis",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QSS 20 — Extension analyses pipeline."
    )
    p.add_argument(
        "--only", nargs="+", choices=list(_ALL_ANALYSES.keys()), default=None,
        metavar="ANALYSIS",
        help=("Run only the listed analyses. Choices: "
              + ", ".join(_ALL_ANALYSES.keys())),
    )
    p.add_argument("--rebuild-panel", action="store_true",
                   help="Force panel rebuild from source files, ignoring cache.")
    return p.parse_args()


def main() -> None:
    args   = _parse_args()
    to_run = set(args.only or _ALL_ANALYSES.keys())

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("=" * 60)
    log.info("QSS 20 — Extension Analyses  (running: %s)", ", ".join(sorted(to_run)))
    log.info("=" * 60)

    # Load base panel from cache (builds if needed)
    panel, ncsl_df = load_or_build_panel(force_rebuild=args.rebuild_panel)
    log.info("Panel: %d rows, %d states.", len(panel), panel["state"].nunique())

    # Load extension data only if the relevant analysis will run
    exp_df    = load_expansion_dates()
    upper_df  = load_ncsl_upper()          if "bicameral"        in to_run else None
    gov_df    = load_governor_party()      if "unified_gov"      in to_run else None
    waiver_df = load_waiver_data(exp_df=exp_df) if "waiver"      in to_run else None

    if "event_study" in to_run:
        log.info("\n--- Analysis 1: Event Study ---")
        run_event_study(panel, ncsl_df)

    if "supermajority_rd" in to_run:
        log.info("\n--- Analysis 2: Supermajority RD ---")
        run_supermajority_rd(panel)

    if "waiver" in to_run:
        log.info("\n--- Analysis 3: Restrictive Waivers ---")
        run_waiver_analysis(panel, waiver_df)

    if "bicameral" in to_run:
        log.info("\n--- Analysis 4: Bicameral Margin ---")
        run_bicameral_analysis(panel, upper_df)

    if "unified_gov" in to_run:
        log.info("\n--- Analysis 5: Unified Government ---")
        run_unified_gov_analysis(panel, gov_df)

    log.info("✓ All extension outputs saved to %s/", OUTPUT_DIR)


if __name__ == "__main__":
    main()
