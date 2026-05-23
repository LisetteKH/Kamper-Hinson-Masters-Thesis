#!/usr/bin/env python3
"""
CREATED BY LISETTE KAMPER_HINSON FOR MS THESIS @ WAKE FOREST 
50_rank_cfr.py
 
Goal
  Rank SKUs from fastest to slowest using batch CFR (good units/hour).
 
What this script does
  1) Reads production.csv.
  2) Computes batch CFR = good_units / run_hours (or derives good_units if needed).
  3) Aggregates by SKU keys and ranks by median CFR.
 
Inputs
  --prod_csv   production.csv
  --out_csv    output ranking CSV
  --sku_keys   columns defining a SKU (default: sku, container_type)
"""
 
from __future__ import annotations
 
import argparse
from pathlib import Path
 
import numpy as np
import pandas as pd
 
 
def safe_percentile(arr: pd.Series, q: float) -> float:
    x = pd.to_numeric(arr, errors="coerce").dropna().to_numpy(dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.nanpercentile(x, q))
 
 
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--sku_keys", nargs="+", default=["sku", "container_type", "color_family", "wc_grade"])
    args = ap.parse_args()
 
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
 
    df = pd.read_csv(args.prod_csv)
 
    if "run_hours" not in df.columns:
        raise ValueError("production.csv must contain 'run_hours' to rank CFR.")
    df["run_hours"] = pd.to_numeric(df["run_hours"], errors="coerce")
 
    if "good_units" in df.columns:
        df["good_units"] = pd.to_numeric(df["good_units"], errors="coerce")
    else:
        if "units_produced" not in df.columns or "yield_loss_frac" not in df.columns:
            raise ValueError("Need 'good_units' OR ('units_produced' and 'yield_loss_frac') in production.csv.")
        units = pd.to_numeric(df["units_produced"], errors="coerce")
        yloss = pd.to_numeric(df["yield_loss_frac"], errors="coerce")
        df["good_units"] = units * (1.0 - yloss).clip(lower=0.0)
 
    for k in args.sku_keys:
        if k not in df.columns:
            df[k] = "UNKNOWN"
 
    df["cfr_uh"] = np.where(df["run_hours"] > 0, df["good_units"] / df["run_hours"], np.nan)
 
    ranked = (
        df.groupby(args.sku_keys, dropna=False)
        .agg(
            median_cfr_uh=("cfr_uh", "median"),
            p10_cfr_uh=("cfr_uh", lambda s: safe_percentile(s, 10)),
            p90_cfr_uh=("cfr_uh", lambda s: safe_percentile(s, 90)),
            batches=("cfr_uh", "count"),
            lines_seen=("line", "nunique") if "line" in df.columns else ("cfr_uh", "count"),
        )
        .reset_index()
        .sort_values("median_cfr_uh", ascending=False)
        .reset_index(drop=True)
    )
    ranked["rank_fastest_first"] = np.arange(1, len(ranked) + 1)
 
    # Ensure container_type column is present for display
    if "container_type" not in ranked.columns:
        ranked["container_type"] = "unknown"
 
    ranked.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} ({len(ranked):,} SKUs)")
 
 
if __name__ == "__main__":
    main()
 
 






