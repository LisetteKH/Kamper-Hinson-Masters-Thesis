#!/usr/bin/env python3
"""
CREATED BY LISETTE KAMPER_HINSON FOR MS THESIS @ WAKE FOREST 
20_compute_capacity_daily.py
 
Produce a daily capacity table by (date, line) combining calendar hours,
downtime, changeovers, and CFR.
 
Two capacity columns:
  capacity_units   — native container units/day  (quarts, gallons, or pails depending on line mix)
  capacity_gallons — gallon-equivalent/day        (cross-line comparable; use for plant totals)
 
This distinction matters because adding quarts + gallons + pails as raw
"units" would be meaningless. Always use capacity_gallons for plant-level
totals and cross-HSD comparisons.
"""
 
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
 
 
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calendar_csv",   default="data/calendar.csv")
    ap.add_argument("--cfr_csv",        default="out/cfr_daily.csv")
    ap.add_argument("--downtime_csv",   default="data/downtime_events.csv")
    ap.add_argument("--changeovers_csv",default="data/changeovers.csv")
    ap.add_argument("--chg_penalty",    type=float, default=1.0)
    ap.add_argument("--out_csv",        default="out/capacity_daily.csv")
    args = ap.parse_args()
 
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
 
    # Calendar
    cal = pd.read_csv(args.calendar_csv)
    cal["date"] = pd.to_datetime(cal["date"], errors="coerce").dt.normalize()
    for col in ["hours_planned", "hours_down", "chg_hours", "sanitation_hours", "break_hours"]:
        if col not in cal.columns:
            cal[col] = 0.0
        cal[col] = pd.to_numeric(cal[col], errors="coerce").fillna(0.0)
 
    # CFR
    cfr = pd.read_csv(args.cfr_csv)
    cfr["date"] = pd.to_datetime(cfr["date"], errors="coerce").dt.normalize()
 
    base = cal.merge(cfr, on=["date", "line"], how="left")
 
    # Optional downtime override
    try:
        dt = pd.read_csv(args.downtime_csv)
        if {"date", "line", "hours_down"}.issubset(dt.columns):
            dt["date"] = pd.to_datetime(dt["date"], errors="coerce").dt.normalize()
            dt_sum = dt.groupby(["date", "line"], as_index=False)["hours_down"].sum()
            base = base.drop(columns=["hours_down"], errors="ignore").merge(
                dt_sum, on=["date", "line"], how="left")
            base["hours_down"] = pd.to_numeric(base["hours_down"], errors="coerce").fillna(0.0)
    except FileNotFoundError:
        pass
 
    # Optional changeover override
    try:
        chg = pd.read_csv(args.changeovers_csv)
        if {"date", "line", "minutes"}.issubset(chg.columns):
            chg["date"] = pd.to_datetime(chg["date"], errors="coerce").dt.normalize()
            chg_sum = chg.groupby(["date", "line"], as_index=False)["minutes"].sum()
            chg_sum["chg_hours"] = (args.chg_penalty *
                pd.to_numeric(chg_sum["minutes"], errors="coerce").fillna(0.0) / 60.0)
            chg_sum = chg_sum.drop(columns=["minutes"])
            base = base.drop(columns=["chg_hours"], errors="ignore").merge(
                chg_sum, on=["date", "line"], how="left")
            base["chg_hours"] = pd.to_numeric(base["chg_hours"], errors="coerce").fillna(0.0)
    except FileNotFoundError:
        pass
 
    # Effective hours
    base["effective_hours"] = (
        base["hours_planned"] - base["hours_down"] - base["chg_hours"]
        - base["sanitation_hours"] - base["break_hours"]
    ).clip(lower=0.0)
 
    # Capacity — native units and gallon-equivalent
    cfr_avg  = pd.to_numeric(base.get("cfr_avg_uh",  pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    cfr_gph  = pd.to_numeric(base.get("cfr_avg_gph", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    eff      = base["effective_hours"].fillna(0.0)
 
    base["capacity_units"]   = cfr_avg * eff   # native container units/day
    base["capacity_gallons"] = cfr_gph  * eff   # gallon-equivalent/day
 
    # Fill missing CFR columns with NA
    for col in ["cfr_avg_uh", "cfr_p10_uh", "cfr_p90_uh",
                "cfr_avg_gph", "cfr_p10_gph", "cfr_p90_gph",
                "n_batches", "run_hours_sum", "good_units_sum", "good_gallons_sum"]:
        if col not in base.columns:
            base[col] = pd.NA
 
    out_cols = [
        "date", "line",
        "hours_planned", "hours_down", "chg_hours", "sanitation_hours", "break_hours",
        "effective_hours",
        "cfr_avg_uh", "cfr_p10_uh", "cfr_p90_uh",
        "cfr_avg_gph", "cfr_p10_gph", "cfr_p90_gph",
        "n_batches", "run_hours_sum", "good_units_sum", "good_gallons_sum",
        "capacity_units",    # native container units/day
        "capacity_gallons",  # gallon-equivalent/day — USE THIS for plant totals
    ]
    out = base[out_cols].sort_values(["line", "date"])
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} ({len(out):,} rows)")
 
    # Sanity print
    plant_gal = out.groupby("date")["capacity_gallons"].sum()
    print(f"  Plant capacity_gallons: median={plant_gal.median():,.0f}  "
          f"P10={plant_gal.quantile(0.1):,.0f}  P90={plant_gal.quantile(0.9):,.0f} gal/day")
    print(f"  Annualized: ~{plant_gal.median()*365/1e6:.2f}M gal/yr")
 
if __name__ == "__main__":
    main()
