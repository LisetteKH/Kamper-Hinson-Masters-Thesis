#!/usr/bin/env python3
"""
CREATED BY LISETTE KAMPER_HINSON FOR MS THESIS @ WAKE FOREST 
99_run_all.py
 
Runs the full capacity planning pipeline end-to-end.
 
Step 0: build_synthetic_data.py  — generates data/ inputs
Step 1: 10_compute_cfr_daily.py  — CFR (good units/hour) by line × day
Step 2: 20_compute_capacity_daily.py — capacity (units/day) using effective hours
Step 3: 30_visualize_metrics.py  — diagnostic plots
Step 4: 40_forecast_capacity.py  — stochastic forecast + plan
Step 5: 50_rank_cfr.py           — SKU speed ranking
 
After each major step, a brief sanity check is printed so you can spot
inflated or implausible numbers immediately.
"""
 
import subprocess
import sys
import pandas as pd
import numpy as np
 
def run(cmd):
    print(f"\n{'─'*60}")
    print("▶  " + " ".join(cmd))
    print('─'*60)
    subprocess.check_call(cmd)
 
 
def sanity_check_cfr():
    try:
        df = pd.read_csv("out/cfr_daily.csv")
        print("\n[SANITY] cfr_daily.csv")
        print(f"  Rows: {len(df):,}")
        for line, g in df.groupby("line"):
            med = g["cfr_avg_uh"].median()
            lo  = g["cfr_avg_uh"].quantile(0.10)
            hi  = g["cfr_avg_uh"].quantile(0.90)
            print(f"  {line}: P10={lo:,.0f}  median={med:,.0f}  P90={hi:,.0f} units/hr")
        print("  Expected: quart HSDs ~15-25 native u/hr · gallon HSDs ~10-20 u/hr · pail HSDs ~3-7 u/hr")
        print("  In gal-equiv/hr: quart ~3-7 · gallon ~10-20 · pail ~15-35")
    except Exception as e:
        print(f"  [WARN] Could not read cfr_daily.csv: {e}")
 
 
def sanity_check_capacity():
    try:
        df = pd.read_csv("out/capacity_daily.csv")
        print("\n[SANITY] capacity_daily.csv")
        print(f"  Rows: {len(df):,}")
        cap_col = "capacity_gallons" if "capacity_gallons" in df.columns else "capacity_units"
        for line, g in df.groupby("line"):
            med = g[cap_col].median()
            lo  = g[cap_col].quantile(0.10)
            hi  = g[cap_col].quantile(0.90)
            print(f"  {line}: P10={lo:,.1f}  median={med:,.1f}  P90={hi:,.1f} gal-equiv/day")
        plant = df.groupby("date")[cap_col].sum()
        print(f"  Plant total: median={plant.median():,.0f}  max={plant.max():,.0f} gal-equiv/day")
        print("  Expected plant daily total: ~3,000–8,000 gal-equiv/day (mid-size plant)")
    except Exception as e:
        print(f"  [WARN] Could not read capacity_daily.csv: {e}")
 
 
def sanity_check_forecast():
    try:
        qdf = pd.read_csv("out/forecast_capacity_quantiles.csv")
        plant_q = qdf.groupby("date")[["q10","q50","q90"]].sum().reset_index()
        print("\n[SANITY] forecast_capacity_quantiles.csv")
        print(f"  Rows: {len(qdf):,} · Horizon days: {qdf['date'].nunique()}")
        print(f"  Plant P10 (avg daily): {plant_q['q10'].mean():,.0f} units")
        print(f"  Plant P50 (avg daily): {plant_q['q50'].mean():,.0f} units")
        print(f"  Plant P90 (avg daily): {plant_q['q90'].mean():,.0f} units")
 
        plan = pd.read_csv("out/planned_capacity_daily.csv")
        plant_plan = plan.groupby("date")["planned_capacity_units"].sum()
        print(f"  Planned (avg daily):   {plant_plan.mean():,.0f} units")
        print("  P10/P50/P90 should be in line with historical capacity_units values above.")
    except Exception as e:
        print(f"  [WARN] Forecast sanity check failed: {e}")
 
 
# Step 0: Synthetic data (used as proof of concept for this dashboard without revealing proprietary real data structure)
run([sys.executable, "build_synthetic_data.py"])
 
# Step 1: CFR 
run([
    sys.executable, "10_compute_cfr_daily.py",
    "--prod_csv", "data/production.csv",
    "--out_csv",  "out/cfr_daily.csv",
])
sanity_check_cfr()
 
# Step 2: Capacity (uses calendar.csv — written by build_synthetic_data.py) 
run([
    sys.executable, "20_compute_capacity_daily.py",
    "--calendar_csv",   "data/calendar.csv",
    "--cfr_csv",        "out/cfr_daily.csv",
    "--downtime_csv",   "data/downtime_events.csv",
    "--changeovers_csv","data/changeovers.csv",
    "--chg_penalty",    "1.0",
    "--out_csv",        "out/capacity_daily.csv",
])
sanity_check_capacity()
 
#  Step 3: Visualizations
run([
    sys.executable, "30_visualize_metrics.py",
    "--cap_csv", "out/capacity_daily.csv",
    "--out_dir", "out/plots",
])
 
# Step 4: Stochastic forecast + plan
run([
    sys.executable, "40_forecast_capacity.py",
    "--cap_csv",            "out/capacity_daily.csv",
    "--horizon_days",       "30",
    "--quantiles",          "0.1,0.5,0.9",
    "--n_scenarios",        "300",
    "--cost_under",         "3.0",
    "--cost_over",          "1.0",
    "--ramp_limit",         "5000",
    "--out_plan_daily_csv", "out/planned_capacity_daily.csv",
    "--out_quantiles_csv",  "out/forecast_capacity_quantiles.csv",
    "--out_scenarios_csv",  "out/capacity_scenarios.csv",
    "--out_csv",            "out/forecast_capacity.csv",
    "--out_plot",           "out/plots/forecast_capacity.png",
])
sanity_check_forecast()
 
# Step 4b: Plant-level rollup 
plan = pd.read_csv("out/planned_capacity_daily.csv", parse_dates=["date"])
plant = plan.groupby("date", as_index=False)["planned_capacity_units"].sum()
plant = plant.rename(columns={"planned_capacity_units": "planned_capacity_plant_units"})
plant.to_csv("out/planned_capacity_plant.csv", index=False)
print(f"\n  planned_capacity_plant.csv: {len(plant)} rows")
 
# Step 5: SKU ranks
run([
    sys.executable, "50_rank_cfr.py",
    "--prod_csv", "data/production.csv",
    "--out_csv",  "out/sku_ranks.csv",
])
 
print("\n" + "═"*60)
print("✓ Pipeline complete. Run: streamlit run app.py")
print("═"*60)
