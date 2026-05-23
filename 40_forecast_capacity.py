#!/usr/bin/env python3
"""
CREATED BY LISETTE KAMPER_HINSON FOR MS THESIS @ WAKE FOREST 
40_forecast_capacity.py
 
Stochastic capacity planning using interpretable time-series methods.
 
Framework alignment (Step 4):
  - Uses exponential smoothing (Holt-Winters) per line — interpretable, low technical debt.
  - Falls back to rolling-median baseline if insufficient history.
  - Uncertainty quantified from historical residuals (empirical, not parametric assumption).
  - Modeled outputs clearly separated from observed values in outputs.
  - All parameters externalized via CLI args (no hard-coded values).
 
Outputs
  - out/forecast_capacity_quantiles.csv   (q10, q50, q90 per date/line)
  - out/capacity_scenarios.csv            (N sampled futures per date/line)
  - out/planned_capacity_daily.csv        (stochastic plan: planned_capacity_units per date/line)
  - out/forecast_capacity.csv             (plant-level rollup; legacy filename)
  - out/plots/planned_capacity_plant.png
"""
 
from __future__ import annotations
 
import argparse
from pathlib import Path
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
 
 
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
 
 
# ─── Exponential smoothing (Holt's linear trend) ────────────────────────────
 
def holt_fit(y: np.ndarray, alpha: float, beta: float) -> tuple[np.ndarray, float, float]:
    """
    Holt's linear (double) exponential smoothing.
    Returns smoothed series, final level, final trend.
    """
    n = len(y)
    L = np.empty(n)
    T = np.empty(n)
    L[0] = y[0]
    T[0] = y[1] - y[0] if n > 1 else 0.0
    for t in range(1, n):
        L[t] = alpha * y[t] + (1 - alpha) * (L[t-1] + T[t-1])
        T[t] = beta  * (L[t] - L[t-1]) + (1 - beta) * T[t-1]
    return L, float(L[-1]), float(T[-1])
 
 
def holt_sse(params: tuple, y: np.ndarray) -> float:
    alpha, beta = params
    if not (0.01 <= alpha <= 0.99 and 0.01 <= beta <= 0.50):
        return 1e12
    L, _, _ = holt_fit(y, alpha, beta)
    return float(np.sum((y - L) ** 2))
 
 
def fit_holt(y: np.ndarray) -> tuple[float, float, float, float]:
    """
    Fit Holt's method by grid search (robust, no gradient issues).
    Returns alpha, beta, final_level, final_trend.
    """
    best_sse   = np.inf
    best_alpha = 0.3
    best_beta  = 0.05
 
    for a in np.arange(0.05, 0.95, 0.10):
        for b in np.arange(0.01, 0.40, 0.05):
            sse = holt_sse((a, b), y)
            if sse < best_sse:
                best_sse   = sse
                best_alpha = a
                best_beta  = b
 
    L, level, trend = holt_fit(y, best_alpha, best_beta)
    return best_alpha, best_beta, level, trend
 
 
def holt_forecast(level: float, trend: float, h: int, damp: float = 0.95) -> np.ndarray:
    """
    Damped-trend forecast for h steps ahead.
    Damping prevents unrealistic trend extrapolation (Step 4 technical debt principle).
    """
    forecasts = np.empty(h)
    phi = damp
    cum_phi = phi
    for i in range(h):
        forecasts[i] = level + cum_phi * trend
        cum_phi *= phi
    return np.maximum(forecasts, 0.0)
 
 
# ─── Per-line modelling ──────────────────────────────────────────────────────
 
def forecast_line(
    history_values: np.ndarray,
    horizon: int,
    n_scenarios: int,
    quantiles: tuple[float, ...],
    min_history: int = 14,
) -> dict:
    """
    Forecast a single line's capacity time series.
 
    Strategy:
    1) If history >= min_history: fit Holt's, get residuals, bootstrap scenarios.
    2) Otherwise: use rolling median + empirical std.
 
    Returns dict with keys: point_forecast, q10, q50, q90, scenarios
    """
    y = np.array(history_values, dtype=float)
    y = y[~np.isnan(y)]
 
    if len(y) == 0:
        zeros = np.zeros(horizon)
        return {
            "point": zeros,
            "quantiles": {q: zeros.copy() for q in quantiles},
            "scenarios": np.zeros((n_scenarios, horizon)),
        }
 
    if len(y) >= min_history:
        alpha, beta, level, trend = fit_holt(y)
        fitted, _, _ = holt_fit(y, alpha, beta)
        residuals = y - fitted
 
        point = holt_forecast(level, trend, horizon)
 
        # Bootstrap residuals to build scenarios
        # Use block bootstrap to preserve autocorrelation structure
        block_len = max(1, len(residuals) // 7)
        scenarios = np.empty((n_scenarios, horizon))
        rng_s = np.random.default_rng(RANDOM_SEED)
        for s in range(n_scenarios):
            noise = np.empty(horizon)
            pos = 0
            while pos < horizon:
                start = int(rng_s.integers(0, max(1, len(residuals) - block_len + 1)))
                block = residuals[start: start + block_len]
                take = min(block_len, horizon - pos)
                noise[pos: pos + take] = block[:take]
                pos += take
            scenarios[s] = np.maximum(point + noise, 0.0)
 
    else:
        # Short history: use median level + empirical spread
        median_level = float(np.median(y))
        std_level    = float(np.std(y)) if len(y) > 1 else median_level * 0.15
        point = np.full(horizon, median_level)
        rng_s = np.random.default_rng(RANDOM_SEED)
        scenarios = np.maximum(
            rng_s.normal(loc=median_level, scale=std_level, size=(n_scenarios, horizon)),
            0.0,
        )
 
    # Quantile bands
    q_dict = {}
    for q in quantiles:
        q_dict[q] = np.quantile(scenarios, q, axis=0)
 
    return {"point": point, "quantiles": q_dict, "scenarios": scenarios}
 
 
# ─── Stochastic plan (newsvendor-style, no external solver) ─────────────────
 
def solve_plan_numpy(
    scenarios: np.ndarray,
    cost_under: float,
    cost_over: float,
    ramp_limit: float | None,
) -> np.ndarray:
    """
    Solve stochastic capacity plan using the analytical newsvendor solution.
 
    For each (day, line), the optimal plan is the (cost_under / (cost_under + cost_over))
    quantile of the scenario distribution — this is the closed-form solution to the
    asymmetric cost minimization, no external solver needed.
 
    Arguments:
      scenarios: shape (n_scenarios, T)
      cost_under: cost per unit of shortfall (plan > realized)
      cost_over:  cost per unit of excess   (realized > plan)
    Returns:
      plan: shape (T,)
    """
    target_quantile = cost_under / (cost_under + cost_over)
    plan = np.quantile(scenarios, target_quantile, axis=0)
    plan = np.maximum(plan, 0.0)
 
    if ramp_limit is not None and ramp_limit > 0:
        # Smooth plan to respect ramp limit (forward pass)
        for t in range(1, len(plan)):
            plan[t] = float(np.clip(plan[t], plan[t-1] - ramp_limit, plan[t-1] + ramp_limit))
 
    return plan
 
 
# ─── Main ────────────────────────────────────────────────────────────────────
 
def main() -> None:
    ap = argparse.ArgumentParser(description="Stochastic capacity forecast (interpretable time-series)")
    ap.add_argument("--cap_csv",            default="out/capacity_daily.csv")
    ap.add_argument("--horizon_days",       type=int,   default=30)
    ap.add_argument("--quantiles",          default="0.1,0.5,0.9")
    ap.add_argument("--n_scenarios",        type=int,   default=200)
    ap.add_argument("--cost_under",         type=float, default=3.0,
                    help="Cost per unit shortfall (plan > realized). Higher → more conservative plan.")
    ap.add_argument("--cost_over",          type=float, default=1.0,
                    help="Cost per unit excess (realized > plan). Higher → more aggressive plan.")
    ap.add_argument("--ramp_limit",         type=float, default=5000.0,
                    help="Max change in planned units per day per line. Set <0 to disable.")
    ap.add_argument("--out_quantiles_csv",  default="out/forecast_capacity_quantiles.csv")
    ap.add_argument("--out_scenarios_csv",  default="out/capacity_scenarios.csv")
    ap.add_argument("--out_plan_daily_csv", default="out/planned_capacity_daily.csv")
    ap.add_argument("--out_csv",            default="out/forecast_capacity.csv")
    ap.add_argument("--out_plot",           default="out/plots/planned_capacity_plant.png")
    args = ap.parse_args()
 
    Path("out/plots").mkdir(parents=True, exist_ok=True)
 
    # ── Load history ──────────────────────────────────────────────────────────
    df = pd.read_csv(args.cap_csv)
    required = {"date", "line"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{args.cap_csv} is missing columns: {missing}")
 
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date", "line", "capacity_units"]).copy()
    # Prefer capacity_gallons for cross-line comparable forecasting
    if "capacity_gallons" in df.columns:
        df["capacity_units"] = pd.to_numeric(df["capacity_gallons"], errors="coerce").fillna(0.0)
    elif "capacity_units" in df.columns:
        df["capacity_units"] = pd.to_numeric(df["capacity_units"], errors="coerce").fillna(0.0)
    else:
        raise ValueError(f"{args.cap_csv} must have capacity_gallons or capacity_units column.")
 
    lines = sorted(df["line"].dropna().unique())
    last_date   = df["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=int(args.horizon_days), freq="D")
 
    quantiles = tuple(sorted(float(x.strip()) for x in args.quantiles.split(",") if x.strip()))
    if not quantiles or any(q <= 0 or q >= 1 for q in quantiles):
        raise ValueError("--quantiles must be strictly between 0 and 1, e.g. 0.1,0.5,0.9")
 
    ramp = None if float(args.ramp_limit) < 0 else float(args.ramp_limit)
 
    # ── Forecast per line ─────────────────────────────────────────────────────
    q_rows, s_rows, p_rows = [], [], []
 
    for line in lines:
        hist_line = (df[df["line"] == line]
                     .sort_values("date")["capacity_units"]
                     .to_numpy(dtype=float))
 
        result = forecast_line(
            history_values=hist_line,
            horizon=int(args.horizon_days),
            n_scenarios=int(args.n_scenarios),
            quantiles=quantiles,
        )
 
        # Build plan for this line
        plan_line = solve_plan_numpy(
            result["scenarios"],
            cost_under=float(args.cost_under),
            cost_over=float(args.cost_over),
            ramp_limit=ramp,
        )
 
        for i, fd in enumerate(future_dates):
            # Quantile rows
            q_row = {"date": fd, "line": line}
            for q in quantiles:
                q_row[f"q{int(q*100):02d}"] = float(result["quantiles"][q][i])
            q_rows.append(q_row)
 
            # Plan rows
            p_rows.append({
                "date": fd,
                "line": line,
                "planned_capacity_units": float(plan_line[i]),
            })
 
        # Scenario rows
        for s in range(int(args.n_scenarios)):
            for i, fd in enumerate(future_dates):
                s_rows.append({
                    "date": fd, "line": line,
                    "scenario": s,
                    "capacity_s": float(result["scenarios"][s, i]),
                })
 
    # ── Write outputs ─────────────────────────────────────────────────────────
    qdf = pd.DataFrame(q_rows).sort_values(["line", "date"])
    qdf.to_csv(args.out_quantiles_csv, index=False)
 
    sdf = pd.DataFrame(s_rows).sort_values(["line", "scenario", "date"])
    sdf.to_csv(args.out_scenarios_csv, index=False)
 
    plan_df = pd.DataFrame(p_rows).sort_values(["line", "date"])
    plan_df.to_csv(args.out_plan_daily_csv, index=False)
 
    # Plant-level rollup (legacy filename)
    plant = plan_df.groupby("date", as_index=False)["planned_capacity_units"].sum()
    plant = plant.rename(columns={"planned_capacity_units": "planned_capacity_plant_units"})
    plant.to_csv(args.out_csv, index=False)
 
    # ── Validation printout ───────────────────────────────────────────────────
    print(f"\nForecast summary (plant total across {int(args.horizon_days)}-day horizon):")
    plant_q = qdf.groupby("date")[["q10", "q50", "q90"]].sum().reset_index() if all(
        c in qdf.columns for c in ["q10","q50","q90"]) else None
    if plant_q is not None:
        p10 = plant_q["q10"].mean()
        p50 = plant_q["q50"].mean()
        p90 = plant_q["q90"].mean()
        print(f"  Avg daily P10 (plant): {p10:,.0f} units")
        print(f"  Avg daily P50 (plant): {p50:,.0f} units")
        print(f"  Avg daily P90 (plant): {p90:,.0f} units")
        avg_plan = plant["planned_capacity_plant_units"].mean()
        print(f"  Avg daily planned:     {avg_plan:,.0f} units")
 
    per_line = plan_df.groupby("line")["planned_capacity_units"].mean()
    print("\nAvg planned units/day per line:")
    for ln, v in per_line.items():
        print(f"  {ln}: {v:,.0f}")
 
    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
 
    # History (observed)
    hist_plant = df.groupby("date", as_index=False)["capacity_units"].sum()
    hist_plant = hist_plant.sort_values("date")
    ax.plot(hist_plant["date"], hist_plant["capacity_units"],
            color="#1B2A4A", linewidth=1.5, label="Observed — gal-equiv/day (historical)", zorder=3)
 
    # Forecast band
    if plant_q is not None:
        ax.fill_between(plant_q["date"], plant_q["q10"], plant_q["q90"],
                        alpha=0.20, color="#2E5590", label="Forecast band (P10–P90)")
        ax.plot(plant_q["date"], plant_q["q50"],
                color="#2E5590", linewidth=1.5, linestyle="--", label="Forecast P50")
 
    # Plan
    ax.plot(plant["date"], plant["planned_capacity_plant_units"],
            color="#C8973A", linewidth=2.0, label="Planned (stochastic optimization)")
 
    # Vertical line separating observed vs forecast
    ax.axvline(x=last_date, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(last_date, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 1,
            " ← history | forecast →", fontsize=9, color="gray")
 
    ax.set_title("Plant Capacity: Observed History + Stochastic Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Capacity (gal-equiv/day — all lines combined)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out_plot, dpi=200)
    plt.close(fig)
 
    print(f"\nWrote:")
    print(f"  {args.out_quantiles_csv}")
    print(f"  {args.out_scenarios_csv}")
    print(f"  {args.out_plan_daily_csv}")
    print(f"  {args.out_csv}")
    print(f"  {args.out_plot}")
 
 
if __name__ == "__main__":
    main()
