"""
01_analysis.py -- Two-way fixed-effects estimates and placebo tests.

INPUT : data/processed/medicaid_panel.csv  (from 00_build.py)
OUTPUT: console tables (the panel estimates reported in the paper's Table 2
        and the within-party split and placebos behind Figs 1 and 4)

Outcome: enrollment ratio (Medicaid+CHIP / <138% FPL population x 100).
SEs are CR1 cluster-robust, clustered by state. Fixed effects (state, year)
are entered as dummies via the ols_cluster helper in utils.py.
"""
import os
import pandas as pd
from utils import ols_cluster, show, PROCESSED_DIR

if __name__ == "__main__":
    panel = pd.read_csv(os.path.join(PROCESSED_DIR, "medicaid_panel.csv"))
    print(f"PANEL: {len(panel)} state-months, {panel.state.nunique()} states")

    # (1) full sample, two-way FE -- the apparent continuous "dial"
    r1 = ols_cluster(panel, "takeup",
                     ["dem_margin", "dem_control", "post_expansion", "mood", "covid"],
                     fe=["state", "year_f"])
    show("(1) Full sample [two-way FE]", r1, ["dem_margin", "dem_control", "post_expansion"])

    # (2) excluding the COVID continuous-enrollment window
    r2 = ols_cluster(panel[panel.covid == 0], "takeup",
                     ["dem_margin", "dem_control", "post_expansion", "mood"],
                     fe=["state", "year_f"])
    show("(2) Excl. COVID", r2, ["dem_margin", "dem_control", "post_expansion"])

    # within-party split (expansion state-months) -- locates the slope (Fig 1)
    exp = panel[panel.expanded == 1]
    rep = ols_cluster(exp[exp.dem_control == 0], "takeup",
                      ["dem_margin", "post_expansion", "mood", "covid"], fe=["state", "year_f"])
    dem = ols_cluster(exp[exp.dem_control == 1], "takeup",
                      ["dem_margin", "post_expansion", "mood", "covid"], fe=["state", "year_f"])
    show("(within) Republican-controlled chambers", rep, ["dem_margin"])
    show("(within) Democratic-controlled chambers", dem, ["dem_margin"])

    # placebos (Fig 4): no expansion population to restrict -> should be ~0
    pa = ols_cluster(panel[panel.expanded == 0], "takeup",
                     ["dem_margin", "mood", "covid"], fe=["state", "year_f"])
    show("(A) Placebo: non-expansion states", pa, ["dem_margin"])

    late = panel[(panel.expanded == 1) &
                 (pd.to_datetime(panel.expansion_date) > pd.Timestamp(2016, 1, 1)) &
                 (panel.post_expansion == 0)]
    pb = ols_cluster(late, "takeup", ["dem_margin", "mood", "covid"], fe=["state", "year_f"])
    show("(B) Placebo: pre-expansion months (late expanders)", pb, ["dem_margin"])

    # robustness: static-2018 denominator
    rs = ols_cluster(panel, "takeup_static",
                     ["dem_margin", "dem_control", "post_expansion", "mood", "covid"],
                     fe=["state", "year_f"])
    show("(R) Static-2018 denominator", rs, ["dem_margin", "dem_control"])
