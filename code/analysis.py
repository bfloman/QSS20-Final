"""
QSS 20 Final Project — Main Analysis Pipeline
Does the margin of Democratic seat share predict Medicaid enrollment?

Author: Ben Floman | Dartmouth College | QSS 20 | May 2026

Data directory structure (place all files in data/ relative to this script):
  data/
    raw_data__2_.csv          ← KFF monthly enrollment (uploaded)
    acs_eligible_pop.csv      ← ACS 2018 population below 138% FPL
    ncsl_lower.csv            ← NCSL lower chamber partisan composition
    expansion_dates.csv       ← Medicaid expansion status and dates
    state_population.csv      ← 2022 Census state population estimates
    presidential_vote.csv     ← Democratic presidential vote share by state/election year
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
KFF_PATH   = os.path.join(DATA_DIR, "raw_data__2_.csv")


def data_path(filename):
    return os.path.join(DATA_DIR, filename)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_kff_enrollment(path=KFF_PATH):
    """
    KFF monthly Medicaid/CHIP enrollment.
    Format: wide, one row per state, columns = 'Mon YYYY__Total Monthly...'
    Source: https://www.kff.org/medicaid/state-indicator/total-medicaid-and-chip-enrollment/
    Returns: state, date (datetime), total_enrollment
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"KFF enrollment file not found at {path}.\n"
            "Download from https://www.kff.org/medicaid/state-indicator/"
            "total-medicaid-and-chip-enrollment/ and save as data/raw_data__2_.csv"
        )

    df = pd.read_csv(path, skiprows=2, encoding="utf-8-sig")
    df = df.rename(columns={df.columns[0]: "state"})
    df = df[~df["state"].isin(["United States"])]
    df = df.drop(columns=[c for c in df.columns if "Footnote" in str(c)], errors="ignore")

    value_cols = [c for c in df.columns if c != "state"]
    df_long = df.melt(id_vars=["state"], value_vars=value_cols,
                      var_name="month_raw", value_name="total_enrollment")

    df_long["month_str"] = df_long["month_raw"].str.split("__").str[0].str.strip()
    df_long["date"] = pd.to_datetime(df_long["month_str"], format="%b %Y", errors="coerce")
    df_long = df_long.dropna(subset=["date"])

    df_long["total_enrollment"] = (
        df_long["total_enrollment"].astype(str)
        .str.replace(",", "", regex=False).str.strip()
        .replace({"N/A": np.nan, "nan": np.nan, "": np.nan})
    )
    df_long["total_enrollment"] = pd.to_numeric(df_long["total_enrollment"], errors="coerce")
    df_long = df_long.dropna(subset=["total_enrollment"])

    return (df_long[["state", "date", "total_enrollment"]]
            .sort_values(["state", "date"]).reset_index(drop=True))


def load_ncsl_lower(path=None):
    """
    NCSL lower chamber partisan composition at 5 election-year snapshots.
    Columns: state, ncsl_year, dem_house_seats, total_house_seats
    Source: NCSL Partisan Composition Database
    Returns: adds dem_share, margin, dem_control
    """
    path = path or data_path("ncsl_lower.csv")
    df = pd.read_csv(path)
    df["dem_share"]   = df["dem_house_seats"] / df["total_house_seats"]
    df["margin"]      = df["dem_share"] - 0.5
    df["dem_control"] = (df["dem_share"] > 0.5).astype(int)
    return df


def load_expansion_dates(path=None):
    """
    Medicaid expansion status and adoption dates by state.
    Columns: state, expanded (0/1), expansion_date (YYYY-MM-DD or blank)
    Source: KFF State Health Facts
    """
    path = path or data_path("expansion_dates.csv")
    df = pd.read_csv(path)
    df["expansion_date"] = df["expansion_date"].replace({np.nan: None, "": None})
    return df


def load_state_population(path=None):
    """
    2022 Census Bureau state population estimates.
    Columns: state, population_2022
    Source: U.S. Census Bureau NST-EST2022-ALLDATA
    """
    path = path or data_path("state_population.csv")
    return pd.read_csv(path)


def load_eligible_population(path=None):
    """
    Population below 138% FPL (Medicaid expansion eligibility threshold).
    Source: ACS 1-year table C17002, annual files 2014-2023.

    Supports two formats:
      Multi-year: columns state, year, eligible_pop (preferred)
        → interpolates linearly across survey years
      Single-year fallback: columns state, eligible_pop_2018
        → holds 2018 value constant (prints a warning)

    Returns state-year panel with columns: state, year, eligible_pop
    """
    path = path or data_path("acs_eligible_pop.csv")
    df = pd.read_csv(path)

    # Multi-year format
    if "year" in df.columns and "eligible_pop" in df.columns:
        study_years = pd.DataFrame({"year": range(2014, 2027)})
        result = df[["state"]].drop_duplicates().merge(study_years, how="cross")
        result = result.merge(df[["state", "year", "eligible_pop"]],
                              on=["state", "year"], how="left")
        result = result.sort_values(["state", "year"])
        result["eligible_pop"] = (
            result.groupby("state")["eligible_pop"]
            .transform(lambda s: s.interpolate(method="linear").ffill().bfill())
        )
        print(f"    Eligible pop: multi-year ACS loaded and interpolated "
              f"for {df['state'].nunique()} states.")
        return result

    # Single-year fallback
    print("    WARNING: Only 2018 ACS estimates found — holding constant. "
          "Replace acs_eligible_pop.csv with multi-year file for best results.")
    rows = []
    for _, r in df.iterrows():
        col = "eligible_pop_2018" if "eligible_pop_2018" in r.index else "eligible_pop"
        for yr in range(2014, 2027):
            rows.append({"state": r["state"], "year": yr, "eligible_pop": r[col]})
    return pd.DataFrame(rows)


def load_presidential_vote(path=None):
    """
    Democratic two-party presidential vote share, linearly interpolated
    between election years for use as public opinion proxy.
    Columns (in file): state, election_year, dem_vote_share
    Source: MIT Election Data and Science Lab; 2024 certified results
    Returns: state-year panel with interpolated mood variable
    """
    path = path or data_path("presidential_vote.csv")
    pres = pd.read_csv(path)
    election_years = sorted(pres["election_year"].unique())
    study_years    = list(range(2014, 2027))
    rows = []

    for state, grp in pres.groupby("state"):
        yr_share = dict(zip(grp["election_year"], grp["dem_vote_share"]))
        for yr in study_years:
            if yr <= election_years[0]:
                mood = yr_share[election_years[0]]
            elif yr >= election_years[-1]:
                mood = yr_share[election_years[-1]]
            else:
                lo = max(e for e in election_years if e <= yr)
                hi = min(e for e in election_years if e >= yr)
                if lo == hi:
                    mood = yr_share[lo]
                else:
                    t = (yr - lo) / (hi - lo)
                    mood = yr_share[lo] + t * (yr_share[hi] - yr_share[lo])
            rows.append({"state": state, "year": yr, "mood": round(mood, 4)})

    print(f"    Presidential vote mood: interpolated for "
          f"{pres['state'].nunique()} states, 2014–2026.")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: PANEL CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

NCSL_YEAR_TO_COVERAGE = {
    2015: ("2015-01-01", "2016-10-31"),
    2017: ("2016-11-01", "2018-10-31"),
    2019: ("2018-11-01", "2020-10-31"),
    2021: ("2020-11-01", "2022-10-31"),
    2023: ("2022-11-01", "2026-01-31"),
}


def merge_ncsl_to_enrollment(enroll_df, ncsl_df):
    """Assign NCSL session to each enrollment month based on session coverage."""
    enroll_df = enroll_df.copy()
    enroll_df["ncsl_year"] = pd.NA

    for ncsl_year, (start, end) in NCSL_YEAR_TO_COVERAGE.items():
        mask = ((enroll_df["date"] >= pd.Timestamp(start)) &
                (enroll_df["date"] <= pd.Timestamp(end)))
        enroll_df.loc[mask, "ncsl_year"] = ncsl_year

    enroll_df = enroll_df.dropna(subset=["ncsl_year"])
    enroll_df["ncsl_year"] = enroll_df["ncsl_year"].astype(int)

    return enroll_df.merge(
        ncsl_df[["state", "ncsl_year", "dem_share", "margin", "dem_control"]],
        on=["state", "ncsl_year"], how="left"
    )


def build_panel(enroll_df, ncsl_df, exp_df, pop_df, mood_df, elig_df=None):
    """
    Build state × month analysis panel.

    Outcome variable:
      If elig_df supplied: takeup_rate = enrollment / eligible_pop * 100
        (enrollment relative to population below 138% FPL)
      Fallback: enroll_rate = enrollment / state_population * 100
    """
    panel = merge_ncsl_to_enrollment(enroll_df, ncsl_df)
    panel["year"] = panel["date"].dt.year

    if elig_df is not None:
        panel = panel.merge(elig_df[["state", "year", "eligible_pop"]],
                            on=["state", "year"], how="left")
        panel = panel.merge(pop_df, on="state", how="left")
        denom = panel["eligible_pop"].fillna(panel["population_2022"])
        panel["enroll_rate"] = panel["total_enrollment"] / denom * 100
        print("    Outcome: Medicaid take-up rate (enrollment / pop below 138% FPL)")
    else:
        panel = panel.merge(pop_df, on="state", how="left")
        panel["enroll_rate"] = panel["total_enrollment"] / panel["population_2022"] * 100
        print("    Outcome: Enrollment rate (enrollment / total population) [fallback]")

    # Expansion status
    panel = panel.merge(exp_df, on="state", how="left")
    panel["expansion_date"] = pd.to_datetime(panel["expansion_date"])
    panel["post_expansion"] = (
        (panel["expanded"] == 1) & (panel["date"] >= panel["expansion_date"])
    ).astype(int)

    # COVID continuous enrollment period
    panel["covid_period"] = (
        (panel["date"] >= pd.Timestamp("2020-03-01")) &
        (panel["date"] <= pd.Timestamp("2023-04-01"))
    ).astype(int)

    # Public opinion (annual)
    panel = panel.merge(mood_df, on=["state", "year"], how="left")
    panel = panel.sort_values(["state", "date"])
    panel["mood"] = panel.groupby("state")["mood"].ffill().bfill()

    panel = panel.dropna(subset=["enroll_rate", "margin", "mood"])
    return panel


def assign_expansion_cohort(panel, exp_df):
    """Assign each state to a Medicaid expansion cohort for visualization."""
    def cohort(row):
        if not row["expanded"]:
            return "Never expanded (pre-2024)"
        d = pd.Timestamp(row["expansion_date"])
        if d.year == 2014:
            return "Expanded 2014"
        elif d.year in [2015, 2016]:
            return "Expanded 2015–2016"
        else:
            return "Expanded 2019–2021"
    exp_df = exp_df.copy()
    exp_df["cohort"] = exp_df.apply(cohort, axis=1)
    return panel.merge(exp_df[["state", "cohort"]], on="state", how="left")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

COHORT_STYLES = {
    "Expanded 2014":             {"color": "#1f77b4", "ls": "-",  "lw": 2.0},
    "Expanded 2015–2016":        {"color": "#ff7f0e", "ls": "--", "lw": 1.8},
    "Expanded 2019–2021":        {"color": "#2ca02c", "ls": "-.", "lw": 1.8},
    "Never expanded (pre-2024)": {"color": "#d62728", "ls": ":",  "lw": 1.6},
}


def fig1_enrollment_by_cohort(panel, outdir=OUTPUT_DIR):
    cohort_monthly = (panel.groupby(["date", "cohort"])["enroll_rate"]
                      .mean().reset_index())
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2023-04-01"),
               color="gray", alpha=0.12, label="COVID continuous enrollment\n(Mar 2020–Apr 2023)")
    for cohort, style in COHORT_STYLES.items():
        sub = cohort_monthly[cohort_monthly["cohort"] == cohort]
        if sub.empty:
            continue
        ax.plot(sub["date"], sub["enroll_rate"],
                color=style["color"], linestyle=style["ls"],
                linewidth=style["lw"], label=cohort)
    ax.set_title("Medicaid Enrollment Over Time by Expansion Cohort\n"
                 "KFF Monthly Data, Jan 2014–Jan 2026", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Mean Medicaid/CHIP Enrollment\n(% of 138% FPL Population)", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = os.path.join(outdir, "fig1_enrollment_by_cohort.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


def fig2_margin_scatter(panel, outdir=OUTPUT_DIR):
    exp_only = panel[panel["post_expansion"] == 1].copy()
    agg = exp_only.groupby(["state", "ncsl_year"]).agg(
        mean_enroll=("enroll_rate", "mean"),
        margin=("margin", "first"),
        dem_control=("dem_control", "first"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(9, 6))
    dm = agg["dem_control"] == 1
    ax.scatter(agg.loc[dm, "margin"],  agg.loc[dm, "mean_enroll"],
               color="#1f77b4", alpha=0.55, s=30, label="Democratic control")
    ax.scatter(agg.loc[~dm, "margin"], agg.loc[~dm, "mean_enroll"],
               color="#d62728", alpha=0.55, s=30, label="Republican control")
    x = agg["margin"].values; y = agg["mean_enroll"].values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() > 2:
        coefs = np.polyfit(x[mask], y[mask], 1)
        x_fit = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(x_fit, np.polyval(coefs, x_fit), "k--", linewidth=1.2,
                label="OLS trend (pooled)")
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Democratic House Seat Margin (Dem Share − 0.5)", fontsize=11)
    ax.set_ylabel("Mean Medicaid Take-Up Rate (%)", fontsize=11)
    ax.set_title("Legislative Margin vs. Medicaid Take-Up Rate\n"
                 "Expansion States Only (Session-Level Averages)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(outdir, "fig2_margin_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


def fig3_flipped_states(panel, ncsl_df, outdir=OUTPUT_DIR):
    control_range = ncsl_df.groupby("state")["dem_control"].agg(["min", "max"])
    flipped = control_range[control_range["min"] != control_range["max"]].index.tolist()
    pf = panel[panel["state"].isin(flipped)]
    monthly = pf.groupby(["date", "state"])["enroll_rate"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2023-04-01"),
               color="gray", alpha=0.10, label="COVID period")
    colors = plt.cm.tab20.colors
    for i, state in enumerate(sorted(flipped)):
        sub = monthly[monthly["state"] == state]
        ax.plot(sub["date"], sub["enroll_rate"],
                color=colors[i % len(colors)], linewidth=1.0,
                alpha=0.75, label=state)
    ax.set_title("Medicaid Take-Up Rate in States That Changed Partisan Control",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Medicaid Take-Up Rate (%)", fontsize=11)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = os.path.join(outdir, "fig3_flipped_states.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: REGRESSION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_regressions(panel, ncsl_df):
    """
    Five two-way fixed-effects specifications.
    Clustered standard errors at state level.
    """
    import statsmodels.formula.api as smf

    control_range = ncsl_df.groupby("state")["dem_control"].agg(["min", "max"])
    flipped = control_range[control_range["min"] != control_range["max"]].index.tolist()

    # post_expansion dropped from expansion-only specs (3, 4) — already conditioned on
    specs = [
        ("(1) Full sample",          panel,
         "margin + dem_control + post_expansion + mood + covid_period"),
        ("(2) Excl. COVID",          panel[panel["covid_period"] == 0],
         "margin + dem_control + post_expansion + mood"),
        ("(3) Expansion states",     panel[panel["post_expansion"] == 1],
         "margin + dem_control + mood + covid_period"),
        ("(4) Exp., excl. COVID",    panel[(panel["post_expansion"]==1) &
                                          (panel["covid_period"]==0)],
         "margin + dem_control + mood"),
        ("(5) Flipped-ctrl. states", panel[panel["state"].isin(flipped)],
         "margin + dem_control + post_expansion + mood + covid_period"),
    ]

    results = {}
    for label, df, rhs in specs:
        df = df.copy().dropna(subset=["enroll_rate","margin","dem_control",
                                       "post_expansion","mood"])
        if len(df) < 50:
            print(f"[SKIP] {label}: insufficient data")
            continue

        # Drop any RHS terms that have zero variance in this subset
        active_terms = []
        for term in rhs.split(" + "):
            col = term.strip()
            if col in df.columns and df[col].std() == 0:
                continue
            active_terms.append(col)
        rhs_clean = " + ".join(active_terms)

        df["state_f"] = df["state"].astype("category")
        df["year_f"]  = df["date"].dt.year.astype("category")

        try:
            model = smf.ols(
                f"enroll_rate ~ {rhs_clean} + C(state_f) + C(year_f)",
                data=df
            ).fit(cov_type="cluster", cov_kwds={"groups": df["state"]})

            key = [t for t in active_terms if t in model.params.index]
            print(f"\n{'='*60}\n{label}\n{'='*60}")
            out = pd.DataFrame({
                "Coef":    model.params[key].round(4),
                "Std Err": model.bse[key].round(4),
                "t-stat":  model.tvalues[key].round(3),
                "P-value": model.pvalues[key].round(4),
            })
            print(out.to_string())
            print(f"N = {int(model.nobs)}, R² (within) = {model.rsquared:.4f}")
            results[label] = model
        except Exception as e:
            print(f"[ERROR] {label}: {e}")

    return results


def save_results_table(results, outdir=OUTPUT_DIR):
    """Export regression results to CSV."""
    key_vars = ["margin", "dem_control", "post_expansion", "mood", "covid_period"]
    rows = []
    for label, res in results.items():
        row = {"Model": label}
        for v in key_vars:
            if v in res.params.index:
                row[f"{v}_coef"] = round(res.params[v], 4)
                row[f"{v}_se"]   = round(res.bse[v], 4)
                row[f"{v}_p"]    = round(res.pvalues[v], 4)
        try:
            row["N"]  = int(res.nobs)
            row["R2"] = round(res.rsquared, 4)
        except Exception:
            pass
        rows.append(row)
    if rows:
        path = os.path.join(outdir, "regression_results.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"\n[SAVED] {path}")


def descriptive_stats(panel, outdir=OUTPUT_DIR):
    """Save descriptive statistics."""
    cols = ["enroll_rate", "margin", "dem_control", "post_expansion",
            "mood", "covid_period"]
    desc = panel[cols].describe().T
    desc.columns = ["N", "Mean", "Std", "Min", "25%", "Median", "75%", "Max"]
    path = os.path.join(outdir, "descriptive_stats.csv")
    desc.round(3).to_csv(path)
    print(f"[SAVED] {path}")
    print("\nDescriptive Statistics:")
    print(desc.round(3).to_string())
    return desc


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("QSS 20 Final Project — Ben Floman")
    print("Does Legislative Margin Predict Medicaid Enrollment?")
    print("=" * 60)

    print("\n[1] Loading data...")
    enroll_df = load_kff_enrollment()
    ncsl_df   = load_ncsl_lower()
    exp_df    = load_expansion_dates()
    pop_df    = load_state_population()
    mood_df   = load_presidential_vote()
    elig_df   = load_eligible_population()

    print(f"    Enrollment rows:  {len(enroll_df):,}")
    print(f"    NCSL rows:        {len(ncsl_df):,}")
    print(f"    States:           {enroll_df['state'].nunique()}")
    print(f"    Date range:       {enroll_df['date'].min().date()} "
          f"to {enroll_df['date'].max().date()}")

    print("\n[2] Building panel...")
    panel = build_panel(enroll_df, ncsl_df, exp_df, pop_df, mood_df,
                        elig_df=elig_df)
    panel = assign_expansion_cohort(panel, exp_df)
    print(f"    Panel rows:       {len(panel):,}")
    print(f"    States in panel:  {panel['state'].nunique()}")

    print("\n[3] Descriptive statistics...")
    descriptive_stats(panel)

    print("\n[4] Generating figures...")
    fig1_enrollment_by_cohort(panel)
    fig2_margin_scatter(panel)
    fig3_flipped_states(panel, ncsl_df)

    print("\n[5] Running regressions...")
    results = run_regressions(panel, ncsl_df)
    save_results_table(results)

    print(f"\n✓ All outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
