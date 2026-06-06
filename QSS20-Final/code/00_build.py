"""
00_build.py -- Clean and merge the four raw sources into the analysis panel.

INPUT : data/raw/*.csv (KFF enrollment, ACS denominator, NCSL composition,
        governor party, presidential vote, expansion dates, restrictive waivers)
OUTPUT: data/processed/medicaid_panel.csv      (state-month analysis panel)
        data/processed/waiver_stateyears.csv   (expansion state-years for 02)

Run first. Prints before/after row counts at each merge for verification.
"""
import os
from utils import build_panel, build_waiver_stateyears, PROCESSED_DIR

if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Building state-month panel ...")
    panel = build_panel()
    panel.to_csv(os.path.join(PROCESSED_DIR, "medicaid_panel.csv"), index=False)

    print("\nBuilding expansion state-year waiver table ...")
    sm = build_waiver_stateyears()
    sm.to_csv(os.path.join(PROCESSED_DIR, "waiver_stateyears.csv"), index=False)
    print(f"  waiver state-years: {len(sm)} ({int(sm.restrictive_waiver.sum())} with active waiver)")

    print(f"\nWrote processed files to {PROCESSED_DIR}")
