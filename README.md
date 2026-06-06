# Direction Without Degree: Party Control, Majority Size, and Medicaid Restriction

Replication repository for the QSS final project. The analysis asks whether the
*size* of a legislative majority (a "dial") shapes Medicaid restriction beyond
the binary fact of which party is in control (a "switch"), using a monthly panel
of 49 states, 2015–2026.

## Repository structure

```
QSS20-Final/
├── README.md
├── requirements.txt
├── code/
│   ├── utils.py             # shared functions (imported by the numbered scripts)
│   ├── 00_build.py          # clean + merge raw sources -> processed panel
│   ├── 01_analysis.py       # two-way fixed-effects estimates + placebos
│   ├── 02_extensions.py     # restriction classification (tree / LOSO CV / L1)
│   └── 03_make_figures.py   # generate the four manuscript figures
├── data/
│   ├── raw/                 # the nine public source files (see Data sources)
│   └── processed/           # built by 00_build.py (medicaid_panel.csv, ...)
├── output/                  # figure PNGs, built by 03_make_figures.py
└── paper/
    └── floman_medicaid.tex  # manuscript (PNAS template)
```

## Quickstart

```bash
pip install -r requirements.txt
cd code
python 00_build.py          # writes data/processed/*.csv
python 01_analysis.py       # prints panel estimates + placebos
python 02_extensions.py     # prints restriction-classification diagnostics
python 03_make_figures.py   # writes output/*.png
```

Run the scripts in numeric order; each later script reads what `00_build.py`
writes. Paths are resolved relative to the repository root, so no paths need to
be edited after cloning.

## Scripts

| Script | Input | What it does | Output |
|---|---|---|---|
| [`code/utils.py`](code/utils.py) | — | Shared functions: `build_panel`, `build_waiver_stateyears`, `ols_cluster` (OLS with CR1 cluster-robust SEs), interpolation and merge helpers. Defines repo-relative paths. | (imported) |
| [`code/00_build.py`](code/00_build.py) | `data/raw/*.csv` | Reshapes KFF enrollment wide→long, builds the ACS poverty denominator, maps NCSL composition snapshots to months, merges governor / opinion / expansion / waiver fields. Prints before/after row counts at every merge. | `data/processed/medicaid_panel.csv` (4,118 state-months); `data/processed/waiver_stateyears.csv` (226 expansion state-years) |
| [`code/01_analysis.py`](code/01_analysis.py) | `data/processed/medicaid_panel.csv` | Two-way fixed-effects regressions (state + year FE, state-clustered SEs): full sample, excl-COVID, within-party split, two placebos, static-denominator robustness. | Console tables (paper Table 2; data behind Figs 1 and 4) |
| [`code/02_extensions.py`](code/02_extensions.py) | `data/processed/waiver_stateyears.csv` | Tests whether majority size predicts restriction beyond direction: depth-3 decision tree, leave-one-state-out cross-validation, L1 (Lasso) logit, drop-Indiana refit, waiver-by-quartile gradient. | Console diagnostics (paper Results "the dial does not survive") |
| [`code/03_make_figures.py`](code/03_make_figures.py) | `data/processed/waiver_stateyears.csv` | Builds the four manuscript figures (cross-validation figure computed live). | `output/fig_withinparty.png`, `fig_waiver_quartile.png`, `fig_crossval.png`, `fig_specforest.png` |

## Data sources

All nine files in `data/raw/` are derived from public sources and are included
in the repo (≈136 KB total); no external download is required.

| File | Source | Unit |
|---|---|---|
| `raw_data__2_.csv` | Kaiser Family Foundation State Health Facts — monthly Medicaid & CHIP enrollment, Jan 2014–Jan 2026 | state × month (wide) |
| `acs_eligible_pop.csv` | U.S. Census ACS 1-year estimates, table C17002 (<138% FPL population), 2014–2023 | state × year |
| `ncsl_lower.csv`, `ncsl_upper.csv` | NCSL Partisan Composition Database (5 election-year snapshots) | state × session |
| `governor_party.csv` | governor party by state and NCSL session | state × session |
| `presidential_vote.csv` | two-party Democratic presidential vote share, 2012–2024 (opinion proxy) | state × election |
| `expansion_dates.csv` | KFF — Medicaid expansion status and adoption date | state |
| `restrictive_waivers.csv` | approved restrictive Section 1115 waivers (work requirements / premiums / caps) | state × year |
| `state_population.csv` | state list / 2022 population | state |

## Notes

- **Standard errors / OLS.** The estimates use a direct numpy implementation of
  OLS with CR1 cluster-robust standard errors (`ols_cluster` in `utils.py`),
  clustered by state. This avoids a heavy dependency and matches an
  `smf.ols(..., cov_type="cluster")` specification.
- **Opinion control.** `00_build.py` uses the two-party presidential-vote share
  as the `mood` proxy. Swap in another state-ideology series under the same
  column name if preferred; results are not sensitive to the choice.
- **NCSL coverage.** No 2025 composition snapshot exists, so the 2023 snapshot
  governs 2023–2026 (`NCSL_COVERAGE` in `utils.py`).
