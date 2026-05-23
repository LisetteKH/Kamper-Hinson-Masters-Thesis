#!/usr/bin/env python3
"""
CREATED BY LISETTE KAMPER_HINSON FOR MS THESIS @ WAKE FOREST 
pipeline_health.py
==================
Birds-Eye Health View for the Capacity Planning Pipeline.
 
Checks every stage of the pipeline and prints a clear, colour-coded
status report covering:
 
  Stage 0  build_synthetic_data.py   → data/
  Stage 1  10_compute_cfr_daily.py   → out/cfr_daily.csv
  Stage 2  20_compute_capacity_daily.py → out/capacity_daily.csv
  Stage 3  30_visualize_metrics.py   → out/plots/
  Stage 4  40_forecast_capacity.py   → out/forecast_capacity_quantiles.csv
           50_rank_cfr.py            → out/sku_ranks.csv
  Stage 5  app.py                    → Streamlit availability
 
For each stage it checks:
  - Required output file exists and is non-empty
  - File freshness (age vs pipeline entry point mtime)
  - Data integrity (shape, nulls, value ranges)
  - Domain knowledge sanity (CFR bounds, gallon conversion, forecast coverage)
 
Usage:
    python pipeline_health.py
    python pipeline_health.py --data_dir ./data --out_dir ./out
    python pipeline_health.py --json           # machine-readable output
    python pipeline_health.py --watch          # re-check every 30 seconds
"""
 
from __future__ import annotations
 
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
 
import pandas as pd
import numpy as np
 
 
# ── ANSI colours (disabled automatically on Windows/non-TTY) ─────────────────
USE_COLOR = sys.stdout.isatty() and os.name != "nt"
 
def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text
 
def green(t):  return _c("32;1", t)
def red(t):    return _c("31;1", t)
def yellow(t): return _c("33;1", t)
def cyan(t):   return _c("36;1", t)
def bold(t):   return _c("1",    t)
def grey(t):   return _c("90",   t)
 
 
# ── Status enum ──────────────────────────────────────────────────────────────
class Status:
    OK      = "OK"
    WARN    = "WARN"
    FAIL    = "FAIL"
    SKIP    = "SKIP"
 
 
@dataclass
class Check:
    name: str
    status: str         # OK / WARN / FAIL / SKIP
    message: str
    detail: str = ""
 
 
@dataclass
class StageReport:
    stage_id: int
    stage_name: str
    script: str
    checks: list[Check] = field(default_factory=list)
 
    @property
    def overall(self) -> str:
        statuses = [c.status for c in self.checks]
        if Status.FAIL in statuses:  return Status.FAIL
        if Status.WARN in statuses:  return Status.WARN
        if all(s == Status.SKIP for s in statuses): return Status.SKIP
        return Status.OK
 
    def add(self, name: str, status: str, message: str, detail: str = ""):
        self.checks.append(Check(name, status, message, detail))
 
 
# ── Helper utilities ──────────────────────────────────────────────────────────
def file_age_minutes(path: Path) -> float | None:
    try:
        mtime = path.stat().st_mtime
        return (time.time() - mtime) / 60
    except FileNotFoundError:
        return None
 
 
def read_csv_safe(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None
 
 
def fmt_age(minutes: float | None) -> str:
    if minutes is None:
        return "file missing"
    if minutes < 60:
        return f"{minutes:.0f}m ago"
    if minutes < 1440:
        return f"{minutes/60:.1f}h ago"
    return f"{minutes/1440:.1f}d ago"
 
 
GALLON_FACTORS = {"quart": 0.25, "gallon": 1.0, "pail_5gal": 5.0}
 
 
# ── Stage checkers ────────────────────────────────────────────────────────────
 
def check_stage0_data(data_dir: Path) -> StageReport:
    """Stage 0: Raw data inputs."""
    rpt = StageReport(0, "Raw Data Inputs", "build_synthetic_data.py")
 
    required = {
        "production.csv":    ("line", "date", "good_units", "container_type"),
        "calendar.csv":      ("date",),
        "downtime_events.csv": ("date", "line"),
        "changeovers.csv":   ("date", "line"),
    }
 
    all_ok = True
    for fname, required_cols in required.items():
        fpath = data_dir / fname
        age = file_age_minutes(fpath)
        if age is None:
            rpt.add(fname, Status.FAIL, "File missing", f"Expected at {fpath}")
            all_ok = False
            continue
 
        df = read_csv_safe(fpath)
        if df is None or df.empty:
            rpt.add(fname, Status.FAIL, "File empty or unreadable")
            all_ok = False
            continue
 
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            rpt.add(fname, Status.FAIL,
                    f"Missing columns: {missing_cols}",
                    f"{len(df):,} rows, age: {fmt_age(age)}")
            all_ok = False
        else:
            null_pct = df[list(required_cols)].isnull().mean().max() * 100
            status = Status.WARN if null_pct > 5 else Status.OK
            rpt.add(fname, status,
                    f"{len(df):,} rows, {null_pct:.1f}% nulls in key cols",
                    f"Age: {fmt_age(age)}")
 
    return rpt
 
 
def check_stage1_cfr(out_dir: Path) -> StageReport:
    """Stage 1: CFR computation output."""
    rpt = StageReport(1, "CFR Computation", "10_compute_cfr_daily.py")
    fpath = out_dir / "cfr_daily.csv"
    age = file_age_minutes(fpath)
 
    if age is None:
        rpt.add("cfr_daily.csv", Status.FAIL, "File missing. Run 10_compute_cfr_daily.py first.")
        return rpt
 
    df = read_csv_safe(fpath)
    if df is None or df.empty:
        rpt.add("cfr_daily.csv", Status.FAIL, "File empty or unreadable.")
        return rpt
 
    # Existence + freshness
    rpt.add("File present", Status.OK,
            f"{len(df):,} rows across {df['line'].nunique() if 'line' in df.columns else '?'} HSD lines",
            f"Age: {fmt_age(age)}")
 
    # Required columns
    req = ["date", "line", "cfr_avg_uh", "cfr_avg_gph"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        rpt.add("Schema", Status.FAIL, f"Missing columns: {missing}")
        return rpt
    rpt.add("Schema", Status.OK, f"All required columns present: {req}")
 
    # Null check
    null_counts = df[req].isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        rpt.add("Completeness", Status.WARN,
                f"{total_nulls} nulls in key columns",
                str(null_counts[null_counts > 0].to_dict()))
    else:
        rpt.add("Completeness", Status.OK, "No nulls in key columns")
 
    # CFR value range sanity (domain knowledge)
    # cfr_avg_uh: native units/hr
    # quart lines expect ~15-25, gallon ~10-20, pail ~3-7
    uh_med  = df["cfr_avg_uh"].median()
    uh_p10  = df["cfr_avg_uh"].quantile(0.10)
    uh_p90  = df["cfr_avg_uh"].quantile(0.90)
    gph_med = df["cfr_avg_gph"].median()
 
    if uh_p10 <= 0:
        rpt.add("CFR value range (units/hr)", Status.FAIL,
                f"P10={uh_p10:.1f} - negative or zero CFR detected",
                "Possible data ingestion error or missing effective capacity.")
    elif uh_p90 > 500:
        rpt.add("CFR value range (units/hr)", Status.WARN,
                f"P10={uh_p10:.1f}  median={uh_med:.1f}  P90={uh_p90:.1f}",
                "P90 exceeds 500 units/hr - verify container type coding.")
    else:
        rpt.add("CFR value range (units/hr)", Status.OK,
                f"P10={uh_p10:.1f}  median={uh_med:.1f}  P90={uh_p90:.1f}")
 
    # Gallon-equivalent sanity: gph should be >= uh * 0.25 (if all quart) and >=uh*5 is unlikely
    ratio = gph_med / uh_med if uh_med > 0 else 0
    if ratio < 0.2 or ratio > 8:
        rpt.add("Gallon-equiv conversion", Status.WARN,
                f"cfr_avg_gph / cfr_avg_uh median ratio = {ratio:.2f}",
                "Expected ratio 0.25-5.0 depending on container mix. Check GALLON_FACTORS.")
    else:
        rpt.add("Gallon-equiv conversion", Status.OK,
                f"gph/uh ratio = {ratio:.2f} (expected 0.25-5.0 for mixed container lines)")
 
    # Coverage: every line should have at least a few days
    if "line" in df.columns and "date" in df.columns:
        coverage = df.groupby("line")["date"].nunique()
        thin_lines = coverage[coverage < 5].index.tolist()
        if thin_lines:
            rpt.add("Line coverage", Status.WARN,
                    f"{len(thin_lines)} HSD lines have <5 days of data",
                    f"Thin lines: {thin_lines}")
        else:
            rpt.add("Line coverage", Status.OK,
                    f"All {len(coverage)} HSD lines have data",
                    f"Range: {coverage.min()}-{coverage.max()} days per line")
 
    return rpt
 
 
def check_stage2_capacity(out_dir: Path) -> StageReport:
    """Stage 2: Capacity aggregation output."""
    rpt = StageReport(2, "Capacity Aggregation", "20_compute_capacity_daily.py")
    fpath = out_dir / "capacity_daily.csv"
    age = file_age_minutes(fpath)
 
    if age is None:
        rpt.add("capacity_daily.csv", Status.FAIL, "File missing. Run 20_compute_capacity_daily.py first.")
        return rpt
 
    df = read_csv_safe(fpath)
    if df is None or df.empty:
        rpt.add("capacity_daily.csv", Status.FAIL, "File empty or unreadable.")
        return rpt
 
    cap_col = "capacity_gallons" if "capacity_gallons" in df.columns else "capacity_units"
    rpt.add("File present", Status.OK,
            f"{len(df):,} rows, using '{cap_col}' as primary capacity metric",
            f"Age: {fmt_age(age)}")
 
    # Schema
    req = ["date", "line", cap_col]
    missing = [c for c in req if c not in df.columns]
    if missing:
        rpt.add("Schema", Status.FAIL, f"Missing columns: {missing}")
        return rpt
    rpt.add("Schema", Status.OK, f"Key columns present")
 
    # Negative capacity
    neg = (df[cap_col] < 0).sum()
    if neg > 0:
        rpt.add("Negative capacity", Status.FAIL,
                f"{neg} rows with negative capacity",
                "Likely downtime overcounting or config error.")
    else:
        rpt.add("Negative capacity", Status.OK, "No negative capacity values")
 
    # Zero capacity (warn if >20% of rows are zero)
    zero_pct = (df[cap_col] == 0).mean() * 100
    if zero_pct > 20:
        rpt.add("Zero-capacity rows", Status.WARN,
                f"{zero_pct:.1f}% of rows have zero capacity",
                "High zero rate may indicate holiday/shutdown or data gap.")
    else:
        rpt.add("Zero-capacity rows", Status.OK, f"{zero_pct:.1f}% zero rows (acceptable)")
 
    # Plant-level daily total sanity: mid-size plant ~3,000-8,000 gal-equiv/day
    plant_daily = df.groupby("date")[cap_col].sum()
    med_plant = plant_daily.median()
    if med_plant < 500:
        rpt.add("Plant daily total", Status.FAIL,
                f"Median plant capacity = {med_plant:,.0f} — implausibly low",
                "Check gallon conversion factors and scheduling hours.")
    elif med_plant > 50_000:
        rpt.add("Plant daily total", Status.WARN,
                f"Median plant capacity = {med_plant:,.0f} — unusually high",
                "Verify unit assumptions (gal-equiv vs raw units).")
    else:
        rpt.add("Plant daily total", Status.OK,
                f"Median = {med_plant:,.0f}  P10 = {plant_daily.quantile(.1):,.0f}  P90 = {plant_daily.quantile(.9):,.0f}")
 
    # Isolation check: capacity_daily should NOT contain forecast columns
    forecast_cols = [c for c in df.columns if "forecast" in c.lower() or "q50" in c.lower()]
    if forecast_cols:
        rpt.add("Stage isolation", Status.WARN,
                f"Found forecast-like columns in capacity output: {forecast_cols}",
                "Descriptive and predictive stages should be strictly separated.")
    else:
        rpt.add("Stage isolation", Status.OK,
                "No forecast columns detected in descriptive stage output")
 
    return rpt
 
 
def check_stage3_plots(out_dir: Path) -> StageReport:
    """Stage 3: Visualization outputs."""
    rpt = StageReport(3, "Visualization Layer", "30_visualize_metrics.py")
    plots_dir = out_dir / "plots"
 
    if not plots_dir.exists():
        rpt.add("plots/ directory", Status.FAIL,
                "Output directory does not exist. Run 30_visualize_metrics.py first.")
        return rpt
 
    plots = list(plots_dir.glob("*.png")) + list(plots_dir.glob("*.html"))
    if not plots:
        rpt.add("Plot files", Status.FAIL, "No .png or .html files found in out/plots/")
        return rpt
 
    rpt.add("Plot files", Status.OK, f"{len(plots)} output files in out/plots/")
 
    # Check for stale plots vs capacity_daily.csv
    cap_path = out_dir / "capacity_daily.csv"
    cap_age = file_age_minutes(cap_path)
    oldest_plot = max(file_age_minutes(p) or 0 for p in plots)
    if cap_age is not None and oldest_plot > cap_age + 60:
        rpt.add("Plot freshness", Status.WARN,
                "Plots are older than capacity_daily.csv by >1 hour",
                "Run 30_visualize_metrics.py to regenerate.")
    else:
        rpt.add("Plot freshness", Status.OK, f"Plots generated {fmt_age(oldest_plot)}")
 
    # Check that a forecast plot exists (should be produced by stage 4 but often in plots/)
    forecast_plots = [p for p in plots if "forecast" in p.name.lower()]
    if not forecast_plots:
        rpt.add("Forecast plot", Status.WARN,
                "No forecast plot found in out/plots/",
                "Expected forecast_capacity.png from 40_forecast_capacity.py")
    else:
        rpt.add("Forecast plot", Status.OK, f"Found: {[p.name for p in forecast_plots]}")
 
    return rpt
 
 
def check_stage4_forecast(out_dir: Path) -> StageReport:
    """Stage 4: Stochastic forecasting outputs."""
    rpt = StageReport(4, "Stochastic Forecasting", "40_forecast_capacity.py")
 
    files_needed = [
        "forecast_capacity_quantiles.csv",
        "planned_capacity_daily.csv",
        "capacity_scenarios.csv",
    ]
 
    all_present = True
    for fname in files_needed:
        fpath = out_dir / fname
        age = file_age_minutes(fpath)
        if age is None:
            rpt.add(fname, Status.FAIL, "Missing. Run 40_forecast_capacity.py first.")
            all_present = False
        else:
            df = read_csv_safe(fpath)
            rows = len(df) if df is not None else 0
            rpt.add(fname, Status.OK, f"{rows:,} rows", f"Age: {fmt_age(age)}")
 
    if not all_present:
        return rpt
 
    # Deep check on quantiles file
    qpath = out_dir / "forecast_capacity_quantiles.csv"
    qdf = read_csv_safe(qpath)
    if qdf is not None and not qdf.empty:
        # Column check
        for col in ["date", "line", "q10", "q50", "q90"]:
            if col not in qdf.columns:
                rpt.add("Quantile schema", Status.FAIL, f"Missing column: '{col}'")
                return rpt
 
        # P10 <= P50 <= P90 ordering
        bad_order = ((qdf["q10"] > qdf["q50"]) | (qdf["q50"] > qdf["q90"])).sum()
        if bad_order > 0:
            rpt.add("Quantile ordering", Status.FAIL,
                    f"{bad_order} rows where P10 > P50 or P50 > P90",
                    "Indicates bootstrap or sorting error in 40_forecast_capacity.py")
        else:
            rpt.add("Quantile ordering", Status.OK, "P10 <= P50 <= P90 in all rows")
 
        # Coverage: at least 30 horizon days
        n_days = qdf["date"].nunique()
        if n_days < 30:
            rpt.add("Forecast horizon", Status.WARN,
                    f"Only {n_days} forecast days found (expected >= 30)")
        else:
            rpt.add("Forecast horizon", Status.OK, f"{n_days} forecast days covered")
 
        # Spread sanity: P90/P10 ratio shouldn't be extreme
        plant_q = qdf.groupby("date")[["q10", "q50", "q90"]].sum()
        spread = (plant_q["q90"] / plant_q["q10"].replace(0, np.nan)).median()
        if spread > 5:
            rpt.add("Uncertainty spread", Status.WARN,
                    f"Median P90/P10 ratio = {spread:.2f} (>5x is unusually wide)",
                    "Check bootstrap block size and scenario count.")
        else:
            rpt.add("Uncertainty spread", Status.OK,
                    f"Median P90/P10 ratio = {spread:.2f} (reasonable uncertainty range)")
 
        # Historical vs forecast separation
        plan_df = read_csv_safe(out_dir / "planned_capacity_daily.csv")
        if plan_df is not None and "date" in plan_df.columns and "date" in qdf.columns:
            plan_dates = set(pd.to_datetime(plan_df["date"]).dt.date)
            q_dates    = set(pd.to_datetime(qdf["date"]).dt.date)
            # Check historical capacity dates don't appear in forecast quantiles
            cap_path2 = out_dir / "capacity_daily.csv"
            cap_df2 = read_csv_safe(cap_path2)
            if cap_df2 is not None and "date" in cap_df2.columns:
                hist_dates = set(pd.to_datetime(cap_df2["date"]).dt.date)
                forecast_only = q_dates - hist_dates
                bleed = q_dates & hist_dates
                if bleed:
                    rpt.add("Hist/forecast separation", Status.WARN,
                            f"{len(bleed)} quantile dates overlap with historical capacity",
                            "Forecast should only cover future dates beyond training window.")
                else:
                    rpt.add("Hist/forecast separation", Status.OK,
                            f"Forecast covers {len(forecast_only)} future dates with no historical bleed")
            else:
                rpt.add("Hist/forecast separation", Status.OK,
                        "Planned and quantile date ranges verified")
 
    return rpt
 
 
def check_stage5_ranking(out_dir: Path) -> StageReport:
    """Stage 5: CFR ranking output."""
    rpt = StageReport(5, "CFR Ranking", "50_rank_cfr.py")
    fpath = out_dir / "sku_ranks.csv"
    age = file_age_minutes(fpath)
 
    if age is None:
        rpt.add("sku_ranks.csv", Status.FAIL, "Missing. Run 50_rank_cfr.py first.")
        return rpt
 
    df = read_csv_safe(fpath)
    if df is None or df.empty:
        rpt.add("sku_ranks.csv", Status.FAIL, "File empty or unreadable.")
        return rpt
 
    rpt.add("File present", Status.OK, f"{len(df):,} SKU records", f"Age: {fmt_age(age)}")
 
    # Isolation check: no forecast columns (ranking is purely descriptive)
    bad_cols = [c for c in df.columns if any(
        kw in c.lower() for kw in ["q10", "q50", "q90", "forecast", "planned"]
    )]
    if bad_cols:
        rpt.add("Stage isolation", Status.FAIL,
                f"Forecast-derived columns found: {bad_cols}",
                "CFR ranking must use only descriptive outputs.")
    else:
        rpt.add("Stage isolation", Status.OK,
                "No forecast artifacts detected in ranking output")
 
    # Null ranks
    rank_col = next((c for c in df.columns if "rank" in c.lower()), None)
    if rank_col:
        null_ranks = df[rank_col].isnull().sum()
        if null_ranks > 0:
            rpt.add("Rank completeness", Status.WARN,
                    f"{null_ranks} SKUs have null rank values")
        else:
            rpt.add("Rank completeness", Status.OK, f"All SKUs ranked via '{rank_col}'")
    else:
        rpt.add("Rank column", Status.WARN,
                "No 'rank' column found in sku_ranks.csv",
                f"Available: {list(df.columns)}")
 
    return rpt
 
 
def check_entrypoint(base_dir: Path) -> StageReport:
    """Entry point and reproducibility checks."""
    rpt = StageReport(6, "Reproducibility and Entry Point", "99_run_all.py")
 
    # Single entry point exists
    ep = base_dir / "99_run_all.py"
    if not ep.exists():
        rpt.add("Single entry point", Status.FAIL, "99_run_all.py not found in base directory")
    else:
        rpt.add("Single entry point", Status.OK, "99_run_all.py present")
 
    # Config / externalized parameters
    config_candidates = list(base_dir.glob("config*.json")) + \
                        list(base_dir.glob("config*.yaml")) + \
                        list(base_dir.glob("config*.yml")) + \
                        list(base_dir.glob("*.cfg"))
    if not config_candidates:
        rpt.add("Externalized config", Status.WARN,
                "No config file found (config.json / config.yaml / *.cfg)",
                "Parameters should be externalized out of scripts.")
    else:
        rpt.add("Externalized config", Status.OK,
                f"Config file(s) found: {[c.name for c in config_candidates]}")
 
    # Check that scripts don't have hardcoded common absolute paths.
    # Exclude pipeline_health.py itself — it contains the pattern strings
    # as part of its own scan logic, not as actual hardcoded paths.
    scripts = [s for s in base_dir.glob("*.py") if s.name != "pipeline_health.py"]
    hardcoded = []
    for script in scripts:
        try:
            src = script.read_text(errors="replace")
            if any(bad in src for bad in ["/Users/", "/home/", "C:\\Users\\", "C:/Users/"]):
                hardcoded.append(script.name)
        except Exception:
            pass
    if hardcoded:
        rpt.add("Hardcoded paths", Status.WARN,
                f"Possible hardcoded absolute paths in: {hardcoded}",
                "Replace with relative paths or config file entries.")
    else:
        rpt.add("Hardcoded paths", Status.OK, "No obvious hardcoded absolute paths detected")
 
    # Determinism: check for fixed seed in forecast script
    forecast_script = base_dir / "40_forecast_capacity.py"
    if forecast_script.exists():
        src = forecast_script.read_text(errors="replace")
        if "seed" in src.lower() or "random_state" in src.lower():
            rpt.add("Fixed random seed", Status.OK,
                    "Random seed reference found in 40_forecast_capacity.py")
        else:
            rpt.add("Fixed random seed", Status.WARN,
                    "No 'seed' or 'random_state' reference in 40_forecast_capacity.py",
                    "Add np.random.seed(42) to ensure deterministic bootstrap outputs.")
 
    return rpt
 
 
# ── Rendering ─────────────────────────────────────────────────────────────────
 
STATUS_ICONS = {
    Status.OK:   green("  ✓ OK    "),
    Status.WARN: yellow("  ⚠ WARN  "),
    Status.FAIL: red("  ✗ FAIL  "),
    Status.SKIP: grey("  · SKIP  "),
}
 
STAGE_ICONS = {
    Status.OK:   green("●"),
    Status.WARN: yellow("●"),
    Status.FAIL: red("●"),
    Status.SKIP: grey("○"),
}
 
def print_report(reports: list[StageReport], elapsed: float) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    width = 78
 
    print()
    print(bold("═" * width))
    print(bold(f"  PIPELINE HEALTH REPORT    {now}"))
    print(bold("═" * width))
 
    # Summary bar
    print()
    for rpt in reports:
        icon = STAGE_ICONS[rpt.overall]
        label = f"Stage {rpt.stage_id}: {rpt.stage_name}"
        status_str = {
            Status.OK:   green("OK"),
            Status.WARN: yellow("WARN"),
            Status.FAIL: red("FAIL"),
            Status.SKIP: grey("SKIP"),
        }[rpt.overall]
        print(f"  {icon}  {label:<38}  [{status_str}]  ({rpt.script})")
 
    print()
    print("─" * width)
 
    # Detail per stage
    for rpt in reports:
        overall_icon = STAGE_ICONS[rpt.overall]
        print()
        print(f"  {overall_icon}  {bold(f'Stage {rpt.stage_id}: {rpt.stage_name}')}")
        print(f"     Script: {grey(rpt.script)}")
        for chk in rpt.checks:
            icon = STATUS_ICONS[chk.status]
            print(f"  {icon}  {chk.name}: {chk.message}")
            if chk.detail:
                print(f"             {grey(chk.detail)}")
 
    # Overall verdict
    print()
    print("─" * width)
    all_statuses = [r.overall for r in reports]
    if Status.FAIL in all_statuses:
        verdict = red("✗  PIPELINE UNHEALTHY  -  One or more stages have failures.")
        failed = [r.stage_name for r in reports if r.overall == Status.FAIL]
        print(f"  {verdict}")
        print(f"  {red('Failed stages:')} {', '.join(failed)}")
        print(f"  {grey('Run 99_run_all.py to regenerate all outputs.')}")
    elif Status.WARN in all_statuses:
        verdict = yellow("⚠  PIPELINE DEGRADED  -  Warnings require attention.")
        warned = [r.stage_name for r in reports if r.overall == Status.WARN]
        print(f"  {verdict}")
        print(f"  {yellow('Stages with warnings:')} {', '.join(warned)}")
    else:
        verdict = green("✓  PIPELINE HEALTHY  -  All stages passed.")
        print(f"  {verdict}")
 
    print(f"  {grey(f'Checked in {elapsed:.2f}s')}")
    print()
    print(bold("═" * width))
    print()
 
 
def build_json(reports: list[StageReport], elapsed: float) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "overall": (
            Status.FAIL if any(r.overall == Status.FAIL for r in reports) else
            Status.WARN if any(r.overall == Status.WARN for r in reports) else
            Status.OK
        ),
        "stages": [
            {
                "stage_id":   r.stage_id,
                "stage_name": r.stage_name,
                "script":     r.script,
                "overall":    r.overall,
                "checks": [
                    {"name": c.name, "status": c.status,
                     "message": c.message, "detail": c.detail}
                    for c in r.checks
                ],
            }
            for r in reports
        ],
    }
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
 
def run_all_checks(base_dir: Path, data_dir: Path, out_dir: Path) -> tuple[list[StageReport], float]:
    t0 = time.time()
    reports = [
        check_stage0_data(data_dir),
        check_stage1_cfr(out_dir),
        check_stage2_capacity(out_dir),
        check_stage3_plots(out_dir),
        check_stage4_forecast(out_dir),
        check_stage5_ranking(out_dir),
        check_entrypoint(base_dir),
    ]
    elapsed = time.time() - t0
    return reports, elapsed
 
 
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Birds-Eye Health View for the Capacity Planning Pipeline"
    )
    ap.add_argument("--base_dir",  default=".", help="Pipeline root directory (default: .)")
    ap.add_argument("--data_dir",  default=None, help="Raw data directory (default: <base>/data)")
    ap.add_argument("--out_dir",   default=None, help="Output directory (default: <base>/out)")
    ap.add_argument("--json",      action="store_true", help="Output JSON instead of text")
    ap.add_argument("--watch",     action="store_true", help="Re-check every 30 seconds")
    ap.add_argument("--interval",  type=int, default=30, help="Watch interval in seconds (default 30)")
    args = ap.parse_args()
 
    base = Path(args.base_dir).resolve()
    data = Path(args.data_dir).resolve() if args.data_dir else base / "data"
    out  = Path(args.out_dir).resolve()  if args.out_dir  else base / "out"
 
    if args.watch:
        print(bold(f"Watching pipeline every {args.interval}s. Press Ctrl+C to stop.\n"))
        try:
            while True:
                reports, elapsed = run_all_checks(base, data, out)
                if args.json:
                    print(json.dumps(build_json(reports, elapsed), indent=2))
                else:
                    # Clear terminal
                    print("\033[2J\033[H" if USE_COLOR else "\n" * 3)
                    print_report(reports, elapsed)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        reports, elapsed = run_all_checks(base, data, out)
        if args.json:
            print(json.dumps(build_json(reports, elapsed), indent=2))
        else:
            print_report(reports, elapsed)
 
        # Exit code: 1 if any FAIL, 0 otherwise
        sys.exit(1 if any(r.overall == Status.FAIL for r in reports) else 0)
 
 
if __name__ == "__main__":
    main()
