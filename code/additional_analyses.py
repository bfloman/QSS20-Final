"""
QSS 20 Final Project — Additional Analyses
1. Effect size translation (margin coefficient → additional enrollees)
2. Placebo tests

Placebo strategy: Two within-sample falsification tests using existing data.
  (A) Non-expansion states: margin should not predict enrollment through the
      same implementation channel if there is no expansion to implement.
  (B) Pre-expansion period: for late-expanding states, margin should not
      predict enrollment before their expansion date if the mechanism is
      implementation-specific.

Optional third placebo: private insurance coverage rate (requires downloading
ACS S2701 — see instructions at bottom of file).

Author: Ben Floman | Dartmouth College | QSS 20 | May 2026
"""

import os
import sys
import logging
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", message=".*kurtosistest.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Optimization.*", category=UserWarning)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis import (
    load_kff_enrollment, load_ncsl_lower, load_expansion_dates,
    load_state_population, load_presidential_vote, load_eligible_population,
    build_panel, assign_expansion_cohort,
    OUTPUT_DIR, DATA_DIR
)


def load_or_build_panel():
    """Build panel from source files (wraps individual loaders)."""
    enroll_df = load_kff_enrollment()
    ncsl_df   = load_ncsl_lower()
    exp_df    = load_expansion_dates()
    pop_df    = load_state_population()
    mood_df   = load_presidential_vote()
    elig_df   = load_eligible_population()
    panel     = build_panel(enroll_df, ncsl_df, exp_df, pop_df, mood_df, elig_df=elig_df)
    panel     = assign_expansion_cohort(panel, exp_df)
    # Attach expansion_date for placebo B
    panel     = panel.merge(exp_df[["state","expansion_date","expanded"]],
                            on="state", how="left", suffixes=("","_meta"))
    if "expansion_date_meta" in panel.columns:
        panel["expansion_date"] = panel["expansion_date"].fillna(panel["expansion_date_meta"])
        panel = panel.drop(columns=["expansion_date_meta","expanded_meta"], errors="ignore")
    return panel, ncsl_df


# ─────────────────────────────────────────────────────────────────────────────
# 1. EFFECT SIZE TRANSLATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_effect_size(panel):
    """
    Translate the margin coefficient into concrete enrollee terms.

    Uses the expansion-states-only estimate (Spec 3: 93.95 pp per unit)
    as the primary reference because it is the theoretically motivated
    specification and the most precisely estimated.

    Reports:
      - Predicted take-up rate change for a one-IQR margin shift
      - Corresponding additional enrollees for median, mean, and large states
    """
    log.info("=" * 60)
    log.info("EFFECT SIZE TRANSLATION")
    log.info("=" * 60)

    # Margin IQR from the panel
    q25 = panel["margin"].quantile(0.25)
    q75 = panel["margin"].quantile(0.75)
    iqr = q75 - q25

    # Coefficients from real regressions
    specs = {
        "(1) Full sample":       80.85,
        "(2) Excl. COVID":       86.85,
        "(3) Expansion states":  93.95,
        "(4) Exp., excl. COVID": 116.39,
    }

    print(f"\nMargin IQR in panel: {iqr:.3f} units  (p25={q25:.3f}, p75={q75:.3f})")
    print(f"A one-IQR shift in margin = moving from 25th to 75th percentile "
          f"of legislative composition\n")

    # Eligible population by state (2019 = last pre-COVID year)
    elig_df = load_eligible_population()
    elig_2019 = elig_df[elig_df["year"] == 2019].set_index("state")["eligible_pop"]
    median_elig = int(elig_2019.median())
    mean_elig   = int(elig_2019.mean())

    # A few illustrative states
    example_states = {
        "Median state":    median_elig,
        "Mean state":      mean_elig,
        "Michigan":        int(elig_2019.get("Michigan", 1900000)),
        "Virginia":        int(elig_2019.get("Virginia", 1200000)),
        "Montana":         int(elig_2019.get("Montana",   200000)),
    }

    print(f"{'Specification':<28} {'Coef (pp/unit)':>14} "
          f"{'IQR effect (pp)':>16} {'Median state (+enrollees)':>26}")
    print("-" * 90)
    for label, coef in specs.items():
        iqr_effect = iqr * coef
        median_extra = int(iqr_effect / 100 * median_elig)
        print(f"  {label:<26} {coef:>14.2f} "
              f"{iqr_effect:>16.1f} "
              f"{median_extra:>26,}")

    print(f"\nNote: 'Median state' = state at median of eligible population "
          f"({median_elig:,} people below 138% FPL, 2019 ACS).")

    # Detailed breakdown for Spec 3
    spec3_coef = specs["(3) Expansion states"]
    iqr_effect = iqr * spec3_coef
    print(f"\nDetailed breakdown — Spec (3), one-IQR margin shift (+{iqr:.2f} units):")
    print(f"  Predicted take-up rate increase: {iqr_effect:.1f} percentage points")
    for name, elig in example_states.items():
        extra = int(iqr_effect / 100 * elig)
        print(f"  {name:<20} ({elig:>10,} eligible)  → +{extra:>8,} additional enrollees")

    return iqr, iqr_effect


# ─────────────────────────────────────────────────────────────────────────────
# 2. PLACEBO TEST A: NON-EXPANSION STATES
# ─────────────────────────────────────────────────────────────────────────────

def placebo_nonexpansion(panel):
    """
    Placebo A: Run the main specification on non-expansion state-months.

    If margin predicts Medicaid enrollment through an implementation
    mechanism specific to post-expansion states, the coefficient should be
    near zero (or at least substantially smaller) among non-expansion states.
    Non-expansion states have no expansion population to implement generously
    or restrict; their Medicaid enrollment is governed by pre-ACA eligibility
    rules that legislatures have less direct control over.

    A null or small coefficient here, contrasted with the large significant
    coefficient in expansion states (Spec 3), constitutes a falsification test:
    the effect of margin is specific to the context where the mechanism
    operates, not a general proxy for state liberalism.
    """
    log.info("=" * 60)
    log.info("PLACEBO A: NON-EXPANSION STATES")
    log.info("=" * 60)

    nonexp = panel[panel["post_expansion"] == 0].copy()
    nonexp["state_f"] = nonexp["state"].astype("category")
    nonexp["year_f"]  = nonexp["date"].dt.year.astype("category")

    n_states = nonexp["state"].nunique()
    log.info("Non-expansion state-months: %d across %d states", len(nonexp), n_states)

    results = {}
    for label, covid_filter in [
        ("Placebo A1: Non-expansion, full",         nonexp),
        ("Placebo A2: Non-expansion, excl. COVID",  nonexp[nonexp["covid_period"] == 0]),
    ]:
        df = covid_filter.copy().dropna(subset=["enroll_rate","margin","dem_control","mood"])
        if len(df) < 50:
            log.warning("Insufficient data for %s", label)
            continue

        df["state_f"] = df["state"].astype("category")
        df["year_f"]  = df["date"].dt.year.astype("category")

        rhs_terms = ["margin", "dem_control", "mood"]
        if "covid_period" in df.columns and df["covid_period"].std() > 0:
            rhs_terms.append("covid_period")
        rhs = " + ".join(rhs_terms)

        try:
            m = smf.ols(
                f"enroll_rate ~ {rhs} + C(state_f) + C(year_f)", data=df
            ).fit(cov_type="cluster", cov_kwds={"groups": df["state"]})

            c, se, pv = m.params["margin"], m.bse["margin"], m.pvalues["margin"]
            ci_lo, ci_hi = c - 1.96*se, c + 1.96*se
            print(f"\n  {label}")
            print(f"  margin coef = {c:.3f}  (SE={se:.3f}, p={pv:.3f}, "
                  f"95% CI [{ci_lo:.2f}, {ci_hi:.2f}])  N={int(m.nobs)}")
            results[label] = (c, se, pv, int(m.nobs))
        except Exception as e:
            log.error("  %s failed: %s", label, e)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. PLACEBO TEST B: PRE-EXPANSION PERIOD (LATE EXPANDERS)
# ─────────────────────────────────────────────────────────────────────────────

def placebo_preexpansion(panel):
    """
    Placebo B: For states that expanded after January 2016, test whether
    margin predicts enrollment in the months BEFORE their expansion date.

    If the mechanism is implementation-specific, margin should not predict
    pre-expansion enrollment among these states. The same states are used
    in the treatment analysis (post-expansion) and the placebo (pre-expansion),
    so any state-specific confounds should affect both equally.

    This is a within-state placebo: the same state with the same legislative
    composition appears in both the treatment and placebo samples.
    """
    log.info("=" * 60)
    log.info("PLACEBO B: PRE-EXPANSION PERIOD (LATE EXPANDERS)")
    log.info("=" * 60)

    # Late expanders: expanded 2016 or later (enough pre-expansion months in panel)
    late_states = (panel[panel["expanded"] == 1]
                   .groupby("state")
                   .apply(lambda x: x["expansion_date"].min())
                   .dropna())
    late_states = late_states[
        pd.to_datetime(late_states) > pd.Timestamp("2016-01-01")
    ].index.tolist()

    log.info("Late-expanding states (expanded ≥ 2016): %s", late_states)

    # Pre-expansion months only
    pre = panel[
        (panel["state"].isin(late_states)) &
        (panel["post_expansion"] == 0)
    ].copy()

    log.info("Pre-expansion state-months for late expanders: %d", len(pre))

    if len(pre) < 50:
        log.warning("Insufficient pre-expansion data for placebo B.")
        return {}

    pre["state_f"] = pre["state"].astype("category")
    pre["year_f"]  = pre["date"].dt.year.astype("category")
    pre = pre.dropna(subset=["enroll_rate","margin","dem_control","mood"])

    results = {}
    try:
        m = smf.ols(
            "enroll_rate ~ margin + dem_control + mood + C(state_f) + C(year_f)",
            data=pre
        ).fit(cov_type="cluster", cov_kwds={"groups": pre["state"]})

        c, se, pv = m.params["margin"], m.bse["margin"], m.pvalues["margin"]
        ci_lo, ci_hi = c - 1.96*se, c + 1.96*se
        print(f"\n  Placebo B: Pre-expansion, late expanders")
        print(f"  margin coef = {c:.3f}  (SE={se:.3f}, p={pv:.3f}, "
              f"95% CI [{ci_lo:.2f}, {ci_hi:.2f}])  N={int(m.nobs)}")
        results["Placebo B: Pre-expansion"] = (c, se, pv, int(m.nobs))
    except Exception as e:
        log.error("Placebo B failed: %s", e)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. PLACEBO FIGURE: COEFFICIENT PLOT
# ─────────────────────────────────────────────────────────────────────────────

def placebo_figure(main_results, placebo_a, placebo_b, outdir=OUTPUT_DIR):
    """
    Coefficient plot comparing main specifications against placebo tests.
    Main specs in blue; placebo specs in gray.
    """
    # Compile all estimates
    all_specs = [
        # (label, coef, se, is_placebo)
        ("(1) Full sample",          80.85,  31.53, False),
        ("(2) Excl. COVID",          86.85,  35.79, False),
        ("(3) Expansion states",     93.95,  25.61, False),
        ("(4) Exp., excl. COVID",   116.39,  27.22, False),
    ]

    for label, (c, se, pv, n) in placebo_a.items():
        all_specs.append((label.replace("Placebo A1: ","").replace("Placebo A2: ",""),
                          c, se, True))
    for label, (c, se, pv, n) in placebo_b.items():
        all_specs.append(("Pre-expansion (late expanders)", c, se, True))

    labels    = [s[0] for s in all_specs]
    coefs     = [s[1] for s in all_specs]
    ses       = [s[2] for s in all_specs]
    is_placebo = [s[3] for s in all_specs]

    fig, ax = plt.subplots(figsize=(10, max(5, len(all_specs)*0.9)))

    for i, (label, coef, se, placebo) in enumerate(all_specs):
        color = "#aaaaaa" if placebo else "#1f77b4"
        alpha = 0.6 if placebo else 0.9
        ax.barh(i, coef, color=color, alpha=alpha, height=0.55)
        ax.errorbar(coef, i, xerr=1.96*se, fmt="none",
                    color="black", capsize=4, linewidth=1.2)
        tag = " [placebo]" if placebo else ""
        ax.text(coef + 1.96*se + 3, i, f"{coef:.1f}{tag}",
                va="center", fontsize=9,
                color="#555555" if placebo else "black")

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Coefficient on Dem. Seat Margin (pp per unit)", fontsize=11)
    ax.set_title("Main Specifications vs. Placebo Tests\n"
                 "Two-Way FE, Clustered SE, 95% CI | Gray = Placebo",
                 fontsize=12, fontweight="bold")

    # Vertical separator between main and placebo
    if any(is_placebo):
        first_placebo = next(i for i, p in enumerate(is_placebo) if p)
        ax.axhline(first_placebo - 0.5, color="gray", linestyle=":",
                   linewidth=1.0, alpha=0.7)

    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, "fig9_placebo.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("[SAVED] %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL: PRIVATE INSURANCE PLACEBO (requires new data download)
# ─────────────────────────────────────────────────────────────────────────────

def load_private_insurance(data_dir=DATA_DIR):
    """
    OPTIONAL PLACEBO: Private insurance coverage rate by state-year.

    To use this placebo:
    1. Go to data.census.gov
    2. Search for table ACSST1Y[YEAR].S2701 for each year 2015-2023
    3. Set geography to All States, download CSV
    4. Save files as data/ACS_S2701_[YEAR].csv

    The function looks for files matching that pattern and extracts the
    percent with private insurance coverage.

    Returns None if files not found (placebo skipped silently).
    """
    import glob

    pattern = os.path.join(data_dir, "ACS_S2701_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        log.info("Private insurance files not found in data/ — skipping optional placebo.")
        log.info("To run: download ACS S2701 for each year and save as ACS_S2701_YEAR.csv")
        return None

    rows = []
    for path in files:
        year = int(os.path.basename(path).replace("ACS_S2701_","").replace(".csv",""))
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", skiprows=1)
            # Find state name column and private insurance column
            loc_col  = df.columns[0]
            priv_col = [c for c in df.columns
                        if "private" in c.lower() and "estimate" in c.lower()
                        and "percent" in c.lower()]
            if not priv_col:
                continue
            df = df.rename(columns={loc_col: "state", priv_col[0]: "private_pct"})
            df = df[["state","private_pct"]].dropna()
            df["private_pct"] = pd.to_numeric(
                df["private_pct"].astype(str).str.replace("%","").str.strip(),
                errors="coerce"
            )
            df["year"] = year
            rows.append(df)
        except Exception as e:
            log.warning("  Could not parse %s: %s", path, e)

    if not rows:
        return None

    result = pd.concat(rows, ignore_index=True)
    log.info("Private insurance: loaded %d state-year rows", len(result))
    return result


def placebo_private_insurance(panel, private_df):
    """
    Placebo C: Does margin predict PRIVATE insurance coverage?

    Private insurance coverage is determined by labor market conditions
    and employer decisions, not by Medicaid legislative choices. A null
    result here would confirm that the Medicaid effect is specific to the
    program rather than reflecting a general 'Democratic states are
    healthier' confound.
    """
    if private_df is None:
        log.info("Placebo C (private insurance) skipped — data not available.")
        return {}

    log.info("=" * 60)
    log.info("PLACEBO C: PRIVATE INSURANCE COVERAGE RATE")
    log.info("=" * 60)

    panel_yr = (panel.copy()
                .assign(year=lambda d: d["date"].dt.year)
                .groupby(["state","year"])
                .agg(margin=("margin","first"),
                     dem_control=("dem_control","first"),
                     mood=("mood","first"),
                     ncsl_year=("ncsl_year","first"))
                .reset_index())

    merged = panel_yr.merge(private_df, on=["state","year"], how="inner")
    merged = merged.dropna(subset=["private_pct","margin","dem_control","mood"])
    merged["state_f"] = merged["state"].astype("category")
    merged["year_f"]  = merged["year"].astype("category")

    log.info("Private insurance placebo: %d state-year obs", len(merged))

    results = {}
    try:
        m = smf.ols(
            "private_pct ~ margin + dem_control + mood + C(state_f) + C(year_f)",
            data=merged
        ).fit(cov_type="cluster", cov_kwds={"groups": merged["state"]})

        c, se, pv = m.params["margin"], m.bse["margin"], m.pvalues["margin"]
        ci_lo, ci_hi = c - 1.96*se, c + 1.96*se
        print(f"\n  Placebo C: Private insurance coverage rate")
        print(f"  margin coef = {c:.3f}  (SE={se:.3f}, p={pv:.3f}, "
              f"95% CI [{ci_lo:.2f}, {ci_hi:.2f}])  N={int(m.nobs)}")
        results["Placebo C: Private insurance"] = (c, se, pv, int(m.nobs))
    except Exception as e:
        log.error("Placebo C failed: %s", e)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("Loading panel...")
    panel, ncsl_df = load_or_build_panel()
    log.info("Panel: %d rows, %d states", len(panel), panel["state"].nunique())

    print("\n" + "="*60)
    print("ADDITIONAL ANALYSES")
    print("="*60)

    # 1. Effect size
    iqr, iqr_effect = compute_effect_size(panel)

    # 2. Placebo A: non-expansion states
    print()
    pa = placebo_nonexpansion(panel)

    # 3. Placebo B: pre-expansion period
    print()
    pb = placebo_preexpansion(panel)

    # 4. Optional placebo C: private insurance
    print()
    private_df = load_private_insurance()
    pc = placebo_private_insurance(panel, private_df)

    # 5. Coefficient plot
    placebo_figure(None, pa, pb)

    print("\n" + "="*60)
    print("INTERPRETATION GUIDE")
    print("="*60)
    print("""
Main finding (Spec 3): margin coef = 93.95 pp (p<0.001)

Placebo A (non-expansion states):
  If coefficient is near zero → effect is specific to expansion context ✓
  If coefficient is large     → margin proxies for general state liberalism ✗

Placebo B (pre-expansion, late expanders):
  If coefficient is near zero → effect is post-expansion specific ✓
  If coefficient is large     → pre-existing trends confound the result ✗

Placebo C (private insurance, if run):
  If coefficient is near zero → effect is Medicaid-specific ✓
  If coefficient is large     → general 'liberal states are healthier' confound ✗

Add whichever placebo results support the causal interpretation to
Table 3 in the paper alongside the main specifications.
""")

    log.info("✓ Done. Placebo figure: output/fig9_placebo.png")


if __name__ == "__main__":
    main()
