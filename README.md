# Legislative Majority Size and Medicaid Enrollment
### Ben Floman | QSS 20 | Dartmouth College | May 2026

Does the *size* of a Democratic legislative majority — not just its direction — independently predict how generously states implement Medicaid? This repository contains all data, code, and output for the paper testing that question using a state-by-month panel from 2015–2026.

**Finding:** Democratic lower chamber seat margin is positive and significant across all four main specifications (p = 0.010 to p < 0.001), with an expansion-states estimate of 93.9 pp per unit of seat share. A one-IQR shift in margin predicts approximately 233,000 additional enrollees in a median-sized state.

---

## Repository Structure

```
QSS20-Final/
├── code/
│   ├── analysis.py              # Main pipeline — builds panel, runs regressions, generates figs 1–3
│   ├── extensions.py            # Five extension analyses — event study, RD, waivers, bicameral, unified gov
│   └── additional_analyses.py   # Effect size translation + placebo tests (fig 9, table 3)
│
├── data/
│   ├── raw_data__2_.csv         # KFF monthly Medicaid/CHIP enrollment (Jan 2014–Jan 2026)
│   ├── acs_eligible_pop.csv     # ACS C17002 population below 138% FPL, annual 2014–2023
│   ├── ncsl_lower.csv           # NCSL lower chamber seat counts, 5 session snapshots
│   ├── ncsl_upper.csv           # NCSL upper chamber seat counts, 5 session snapshots
│   ├── expansion_dates.csv      # Medicaid expansion status and adoption dates by state
│   ├── state_population.csv     # 2022 Census state population estimates (fallback denominator)
│   ├── presidential_vote.csv    # Democratic two-party presidential vote share by state, 2012–2024
│   ├── governor_party.csv       # Governor party (D/R) at each NCSL session snapshot
│   └── restrictive_waivers.csv  # State-years with active restrictive Medicaid waivers
│
├── output/
│   ├── floman_medicaid_margin.pdf   # Compiled paper
│   ├── floman_medicaid_margin.tex   # LaTeX source
│   ├── references.bib               # BibTeX references
│   ├── fig1_enrollment_by_cohort.png
│   ├── fig2_margin_scatter.png
│   ├── fig3_flipped_states.png
│   ├── fig4_event_study.png
│   ├── fig5_supermajority_rd.png
│   ├── fig6_waiver_by_margin.png
│   ├── fig7_bicameral.png
│   ├── fig8_unified_gov.png
│   ├── fig9_placebo.png
│   ├── descriptive_stats.csv
│   └── regression_results.csv
│
└── README.md
```

---

## How to Run

Scripts expect to be run from the repo root. Data is read from `data/` and output is written to `output/`.

```bash
cd QSS20-Final

# Main analysis (builds panel, figs 1–3, regression table)
python code/analysis.py

# Extension analyses (figs 4–8)
python code/extensions.py

# Effect size + placebo tests (fig 9, table 3)
python code/additional_analyses.py

# Run a single extension
python code/extensions.py --only waiver
python code/extensions.py --only bicameral event_study

# Force panel rebuild (e.g. after updating data files)
python code/analysis.py --rebuild-panel
```

Output goes to `output/`. The panel is cached to `output/panel_cache.parquet` — delete it to force a rebuild.

---

## Data Sources

| File | Source | Notes |
|------|--------|-------|
| `raw_data__2_.csv` | [KFF State Health Facts](https://www.kff.org/medicaid/state-indicator/total-medicaid-and-chip-enrollment/) | Monthly enrollment, Jan 2014–Jan 2026. Download → all states, monthly, CSV. |
| `acs_eligible_pop.csv` | [data.census.gov](https://data.census.gov), table ACSDT1Y[YEAR].C17002 | Annual 1-year estimates 2014–2023; 5-year used for 2020 (Census suspended 1-year due to COVID). Eligible pop = Under .50 + .50–.99 + 1.00–1.24 + 0.54 × (1.25–1.49) buckets. |
| `ncsl_lower.csv` / `ncsl_upper.csv` | [NCSL Partisan Composition Database](https://www.ncsl.org/elections-and-campaigns/partisan-composition) | Official NCSL figures, verified against historical records. 5 snapshots: Jan 2015, Mar 2017, Apr 2019, Feb 2021, Feb 2023. Nebraska excluded (unicameral nonpartisan). |
| `expansion_dates.csv` | [KFF State Health Facts](https://www.kff.org/affordable-care-act/state-indicator/state-activity-around-expanding-medicaid-under-the-affordable-care-act/) | Expansion status and adoption dates. |
| `presidential_vote.csv` | [MIT MEDSL](https://doi.org/10.7910/DVN/42MVDX); 2024 certified state returns | Democratic two-party vote share at presidential election years; interpolated linearly between elections in the pipeline. |
| `governor_party.csv` | NCSL Partisan Composition Database | Governor party (1=D, 0=R) at each session snapshot. Independents coded 0. |
| `restrictive_waivers.csv` | KFF Medicaid Work Requirements Tracker; CMS approval records | State-years with active approved restrictive waivers (work requirements, premiums, enrollment caps). Sparse format — only rows where waiver is active. |
| `state_population.csv` | U.S. Census Bureau NST-EST2022-ALLDATA | Used as fallback denominator when ACS eligible pop is missing for a state-year. |

---

## Key Variables

**Outcome:** `enroll_rate` — Medicaid/CHIP enrollment as % of annual ACS population below 138% FPL. Rates exceed 100% because total enrollment includes children (CHIP), elderly, and disabled populations beyond the ACA expansion pool.

**Key independent variable:** `margin` — Democratic lower chamber seat share minus 0.5. Positive = Democratic majority; negative = Republican majority.

**Panel structure:** State × month, 4,118 observations, 49 states (Nebraska excluded). NCSL session coverage begins January 2015; January–December 2014 enrollment data excluded.

**Session assignment:** Each NCSL snapshot is matched to enrollment months from the following November election through the subsequent October.

---

## Replication Notes

- The `load_eligible_population()` function automatically detects whether `acs_eligible_pop.csv` is multi-year (preferred) or single-year (fallback) format and handles both.
- Regressions use `statsmodels` OLS with state and year dummy variables and clustered standard errors. The `linearmodels` package is not required.
- The panel cache (`output/panel_cache.parquet`) speeds up repeated runs. Delete it to force a full rebuild from source files.
- 133 state-month observations use `state_population.csv` as the denominator fallback (states with missing ACS coverage in a given year). These are flagged in the build log.

---

## Dependencies

```bash
pip install pandas numpy matplotlib scipy statsmodels openpyxl pyarrow
```

Python 3.9+ required.

---

## Paper

The compiled paper (`output/floman_medicaid_margin.pdf`) can be recompiled from source on any machine with LaTeX installed:

```bash
cd output
pdflatex floman_medicaid_margin.tex
bibtex floman_medicaid_margin
pdflatex floman_medicaid_margin.tex
pdflatex floman_medicaid_margin.tex
```

Or upload `floman_medicaid_margin.tex`, `references.bib`, and all `fig*.png` files to [Overleaf](https://www.overleaf.com) for browser-based compilation.

---

## Citation

Floman, Ben. 2026. "Legislative Majority Size and Medicaid Enrollment: Evidence from a State Panel, 2015–2026." QSS 20 Final Paper, Dartmouth College.
