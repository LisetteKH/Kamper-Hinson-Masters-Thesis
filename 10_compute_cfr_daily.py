#!/usr/bin/env python3
"""
CREATED BY LISETTE KAMPER_HINSON FOR MS THESIS @ WAKE FOREST

10_compute_cfr_daily.py
 
Compute daily CFR by (date, line) from batch-level production data.
 
Outputs two CFR metrics per line-day:
  cfr_avg_uh    — median native units/hr  (quarts/hr, gallons/hr, or pails/hr depending on mix)
  cfr_avg_gph   — median gallon-equivalent/hr  (cross-line comparable)
 
Gallon conversion:
  quart=0.25, gallon=1.0, pail_5gal=5.0


"""
 
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
 
GALLONS_PER_UNIT = {"quart": 0.25, "gallon": 1.0, "pail_5gal": 5.0}
 
def safe_percentile(s: pd.Series, q: float) -> float:
    arr = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanpercentile(arr, q))
 
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod_csv", default="data/production.csv")
    ap.add_argument("--out_csv",  default="out/cfr_daily.csv")
    args = ap.parse_args()
 
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
 
    df = pd.read_csv(args.prod_csv)
    if "date" not in df.columns or "line" not in df.columns:
        raise ValueError("production.csv must include 'date' and 'line' columns.")
 
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
 
    # Good units (native container units)
    if "good_units" in df.columns:
        df["good_units"] = pd.to_numeric(df["good_units"], errors="coerce")
    else:
        if "units_produced" not in df.columns or "yield_loss_frac" not in df.columns:
            raise ValueError("Need 'good_units' OR ('units_produced' + 'yield_loss_frac').")
        units = pd.to_numeric(df["units_produced"], errors="coerce")
        yloss = pd.to_numeric(df["yield_loss_frac"], errors="coerce")
        df["good_units"] = units * (1.0 - yloss).clip(lower=0.0)
 
    if "run_hours" not in df.columns:
        raise ValueError("production.csv must include 'run_hours'.")
    df["run_hours"] = pd.to_numeric(df["run_hours"], errors="coerce")
 
    # Gallon-equivalent per batch
    if "good_gallons" in df.columns:
        df["good_gallons"] = pd.to_numeric(df["good_gallons"], errors="coerce")
    else:
        # Derive from container_type if available, else assume gallon
        if "container_type" in df.columns:
            df["_gfactor"] = df["container_type"].map(GALLONS_PER_UNIT).fillna(1.0)
        else:
            df["_gfactor"] = 1.0
        df["good_gallons"] = df["good_units"] * df["_gfactor"]
 
    # CFR: native units/hr and gal-equiv/hr per batch
    df["cfr_good_uh"]  = np.where(df["run_hours"] > 0, df["good_units"]  / df["run_hours"], np.nan)
    df["cfr_good_gph"] = np.where(df["run_hours"] > 0, df["good_gallons"] / df["run_hours"], np.nan)
 
    daily = (
        df.groupby(["date", "line"], dropna=False)
        .agg(
            cfr_avg_uh   =("cfr_good_uh",  "median"),
            cfr_p10_uh   =("cfr_good_uh",  lambda s: safe_percentile(s, 10)),
            cfr_p90_uh   =("cfr_good_uh",  lambda s: safe_percentile(s, 90)),
            cfr_avg_gph  =("cfr_good_gph", "median"),   # gallons/hr — cross-line comparable
            cfr_p10_gph  =("cfr_good_gph", lambda s: safe_percentile(s, 10)),
            cfr_p90_gph  =("cfr_good_gph", lambda s: safe_percentile(s, 90)),
            n_batches    =("cfr_good_uh",  "count"),
            run_hours_sum=("run_hours",    "sum"),
            good_units_sum  =("good_units",   "sum"),
            good_gallons_sum=("good_gallons", "sum"),
        )
        .reset_index()
        .sort_values(["line", "date"])
    )
 
    daily.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} ({len(daily):,} rows)")
 
    # Sanity print
    for line, g in daily.groupby("line"):
        med_uh  = g["cfr_avg_uh"].median()
        med_gph = g["cfr_avg_gph"].median()
        print(f"  {line}: median CFR = {med_uh:.1f} native u/hr  |  {med_gph:.2f} gal-equiv/hr")
 
if __name__ == "__main__":
    main()
