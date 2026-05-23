# CREATED BY LISETTE KAMPER_HINSON FOR MS THESIS @ WAKE FOREST 
import pandas as pd
import numpy as np
import streamlit as st
import datetime as dt
 
from utils import (
    rollups,
    apply_what_if_to_drivers,
    make_future_driver_rows_from_history,
    quantile_band_from_adjusted_drivers,
    sample_scenarios_from_quantiles,
    solve_stochastic_plan_ortools,
    compute_efficiency_gap,
    compute_cfr_trend,
    container_display_name,
    line_unit_label,
)
 
try:
    import holidays as holidays_pkg
except Exception:
    holidays_pkg = None
 
try:
    import altair as alt
except Exception:
    alt = None
 
# ─── PAGE CONFIG ─────────
st.set_page_config(
    page_title="Plant Capacity Planner",
    layout="wide",
    initial_sidebar_state="collapsed",
)
 
# ─── GLOBAL STYLES ───────
st.markdown(
    """
    <style>
    /* Hide number input steppers */
    div[data-baseweb="input"] button { display: none !important; }
 
    /* Card styling */
    .kpi-card {
        background: #1B2A4A;
        border-radius: 10px;
        padding: 18px 22px 14px 22px;
        text-align: center;
        color: white;
        margin-bottom: 4px;
    }
    .kpi-card .kpi-label {
        font-size: 12px;
        color: #ADBDE0;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .kpi-card .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.1;
    }
    .kpi-card .kpi-sub {
        font-size: 12px;
        color: #C8973A;
        margin-top: 4px;
    }
 
    /* Status badge */
    .status-green  { background:#1a4731; color:#4ade80; padding:5px 14px; border-radius:20px; font-size:13px; font-weight:600; display:inline-block; }
    .status-yellow { background:#3d3008; color:#fbbf24; padding:5px 14px; border-radius:20px; font-size:13px; font-weight:600; display:inline-block; }
    .status-red    { background:#450a0a; color:#f87171; padding:5px 14px; border-radius:20px; font-size:13px; font-weight:600; display:inline-block; }
 
    /* Section header bar */
    .section-header {
        background: #1B2A4A;
        color: #C8973A;
        padding: 8px 16px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 12px;
        margin-top: 8px;
    }
 
    /* Step pill */
    .step-pill {
        background: #2E5590;
        color: white;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 14px;
        margin-right: 8px;
    }
 
    /* Provenance banner */
    .provenance-banner {
        background: #F0F4FF;
        border-left: 4px solid #2E5590;
        padding: 10px 16px;
        border-radius: 0 6px 6px 0;
        font-size: 12px;
        color: #333;
        margin: 8px 0 16px 0;
    }
 
    /* Forecast vs observed distinction */
    .legend-observed   { display:inline-block; width:14px; height:4px; background:#1B2A4A; border-radius:2px; margin-right:5px; vertical-align:middle; }
    .legend-forecast   { display:inline-block; width:14px; height:4px; background:#C8973A; border-radius:2px; border-top:2px dashed #C8973A; margin-right:5px; vertical-align:middle; }
    .legend-band       { display:inline-block; width:14px; height:10px; background:rgba(94,143,204,0.25); border-radius:2px; margin-right:5px; vertical-align:middle; }
 
    /* Tab overrides */
    button[data-baseweb="tab"] { font-size: 15px !important; font-weight: 600 !important; }
 
    /* Tighten metric labels */
    [data-testid="metric-container"] label { font-size: 12px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)
 
# ─── DATE HELPERS ────────
def today_norm() -> pd.Timestamp:
    return pd.Timestamp.today().normalize()
 
def planning_window(start_date: dt.date, horizon_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = start_ts + pd.Timedelta(days=int(horizon_days) - 1)
    return start_ts, end_ts
 
def filter_to_window(df: pd.DataFrame, start_ts, end_ts, date_col="date") -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
    return out[(out[date_col] >= start_ts) & (out[date_col] <= end_ts)]
 
# ─── DATA LOADING ────────
@st.cache_data
def load_outputs():
    cap = pd.read_csv("out/capacity_daily.csv", parse_dates=["date"])
    cap["date"] = pd.to_datetime(cap["date"]).dt.normalize()
    q = plan = plant = None
    try:
        q = pd.read_csv("out/forecast_capacity_quantiles.csv", parse_dates=["date"])
        q["date"] = pd.to_datetime(q["date"]).dt.normalize()
    except Exception:
        pass
    try:
        plan = pd.read_csv("out/planned_capacity_daily.csv", parse_dates=["date"])
        plan["date"] = pd.to_datetime(plan["date"]).dt.normalize()
    except Exception:
        pass
    try:
        plant = pd.read_csv("out/forecast_capacity.csv", parse_dates=["date"])
        plant["date"] = pd.to_datetime(plant["date"]).dt.normalize()
    except Exception:
        pass
    return cap, q, plan, plant
 
def daily_weekly_monthly_series(df, date_col, value_col):
    daily = df[[date_col, value_col]].copy().sort_values(date_col)
    weekly, monthly = rollups(
        daily.rename(columns={date_col: "date", value_col: "value"}),
        date_col="date", value_col="value",
    )
    weekly = weekly.rename(columns={"date": date_col, "value": value_col})
    monthly = monthly.rename(columns={"date": date_col, "value": value_col})
    return daily, weekly, monthly

def _plan_gal_col(df: pd.DataFrame) -> str:
    """Return the best gallons column from a plan DataFrame."""
    if "planned_capacity_gallons" in df.columns:
        return "planned_capacity_gallons"
    return "planned_capacity_units"   # fallback for old data

def _qdf_gal_cols(qdf: pd.DataFrame) -> tuple[str, str, str]:
    """Return q10/q50/q90 column names, preferring gallon variants."""
    if "q10_gal" in qdf.columns:
        return "q10_gal", "q50_gal", "q90_gal"
    return "q10", "q50", "q90"
 
# ─── HELPERS ─────────────
def safe_median(series):
    try:
        s = pd.to_numeric(series, errors="coerce")
        v = float(np.nanmedian(s.values))
        return None if np.isnan(v) else v
    except Exception:
        return None
 
def safe_ratio(new, old, default=1.0):
    try:
        if old is None or old == 0 or np.isnan(old): return float(default)
        if new is None or np.isnan(new): return float(default)
        return float(new) / float(old)
    except Exception:
        return float(default)
 
def _safe_div(a, b):
    try:
        return float(a) / float(b) if b and b > 0 else np.nan
    except Exception:
        return np.nan
 
def _risk_flag_from_ratio(x):
    if pd.isna(x): return "—"
    if x >= 1.10: return "🔴 High risk"
    if x >= 1.02: return "🟡 Moderate"
    return "🟢 On track"
 
def build_rolling_totals(daily_df, date_col, value_col):
    d = daily_df[[date_col, value_col]].copy().sort_values(date_col)
    d[date_col] = pd.to_datetime(d[date_col]).dt.normalize()
    d = d.set_index(date_col)
    roll7  = d.rolling("7D").sum().reset_index().rename(columns={value_col: f"{value_col}_roll7"})
    roll30 = d.rolling("30D").sum().reset_index().rename(columns={value_col: f"{value_col}_roll30"})
    return roll7, roll30
 
def plant_stress_indicator(total_plan_units, total_q50_units, total_q10_units):
    if total_q50_units is None or total_q50_units <= 0:
        return "—", None
    ratio = float(total_plan_units) / float(total_q50_units)
    if ratio >= 1.10: return "🔴 High stress", ratio
    if ratio >= 1.02: return "🟡 Moderate stress", ratio
    if total_q10_units is not None and total_q10_units > 0 and total_plan_units < total_q10_units:
        return "🟢 Very conservative", ratio
    return "🟢 Low stress", ratio
 
def allocate_capacity_by_container_from_sku_ranks(total_units, sku_ranks_df, use_speed_col="median_cfr_uh"):
    ranks = sku_ranks_df.copy()
    ranks[use_speed_col] = pd.to_numeric(ranks[use_speed_col], errors="coerce")
    ranks = ranks.dropna(subset=[use_speed_col, "container_type"])
    ranks = ranks[ranks[use_speed_col] > 0]
    if ranks.empty or total_units <= 0:
        return pd.DataFrame(columns=["container_type", "planned_units", "share_pct"])
    speed_sum = float(ranks[use_speed_col].sum())
    ranks["weight"] = ranks[use_speed_col] / speed_sum
    ranks["allocated_units"] = float(total_units) * ranks["weight"]
    out = (ranks.groupby("container_type", as_index=False)["allocated_units"]
           .sum().rename(columns={"allocated_units": "planned_units"})
           .sort_values("planned_units", ascending=False))
    out["share_pct"] = 100.0 * out["planned_units"] / float(out["planned_units"].sum())
    return out
 
def slow_sku_dominance_flag(sku_ranks_df, speed_col="median_cfr_uh", slow_quantile=0.25, warn_threshold_pct=40.0):
    ranks = sku_ranks_df.copy()
    ranks[speed_col] = pd.to_numeric(ranks[speed_col], errors="coerce")
    ranks = ranks.dropna(subset=[speed_col])
    ranks = ranks[ranks[speed_col] > 0]
    if ranks.empty: return 0.0, "—"
    ranks["weight"] = ranks[speed_col] / float(ranks[speed_col].sum())
    cutoff = float(ranks[speed_col].quantile(slow_quantile))
    slow = ranks[ranks[speed_col] <= cutoff]
    slow_share = 100.0 * float(slow["weight"].sum())
    if slow_share >= warn_threshold_pct: return slow_share, "🔴 Slow SKUs dominate mix"
    if slow_share >= warn_threshold_pct * 0.75: return slow_share, "🟡 Slow SKUs notable"
    return slow_share, "🟢 Mix looks healthy"
 
def default_weekly_schedule_normal_week():
    return pd.DataFrame({
        "Day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        "Shifts per day": [3, 3, 3, 3, 3, 2, 2],
        "Hours per shift": [8.0, 8.0, 8.0, 8.0, 8.0, 12.0, 12.0],
    })
 
def weekly_schedule_stats(weekly_df):
    w = weekly_df.copy()
    w["planned_hours"] = (pd.to_numeric(w["Shifts per day"], errors="coerce").fillna(0.0)
                          * pd.to_numeric(w["Hours per shift"], errors="coerce").fillna(0.0))
    planned_by_day = dict(zip(w["Day"].astype(str).tolist(), w["planned_hours"].tolist()))
    avg_planned = float(np.nanmean(w["planned_hours"].values)) if len(w) else 0.0
    avg_shifts  = float(np.nanmean(pd.to_numeric(w["Shifts per day"], errors="coerce").fillna(0.0).values)) if len(w) else 0.0
    return planned_by_day, avg_planned, avg_shifts
 
def ensure_session_defaults():
    st.session_state.setdefault("initialized", True)
    st.session_state.setdefault("horizon_days", 30)
    st.session_state.setdefault("scenario_quality", "Standard (200 futures)")
    st.session_state.setdefault("preset", "Normal week")
    st.session_state.setdefault("plan_start_date", today_norm().date())
    st.session_state.setdefault("show_history_overview", False)
    st.session_state.setdefault("show_history_plan_explorer", False)
    st.session_state.setdefault("closed_dates", [])
    if "weekly_schedule_df" not in st.session_state:
        st.session_state.weekly_schedule_df = default_weekly_schedule_normal_week()
    st.session_state.setdefault("daily_override_df", None)
    st.session_state.setdefault("downtime_min_per_shift", 30)
    st.session_state.setdefault("changeovers_per_day", 3.0)
    st.session_state.setdefault("avg_changeover_minutes", 45)
    st.session_state.setdefault("eff_percent", 100.0)
    st.session_state.setdefault("plan_style", "Balanced")
    st.session_state.setdefault("ramp_on", True)
    st.session_state.setdefault("ramp_limit", 5000.0)
    st.session_state.setdefault("show_holiday_markers", True)
    st.session_state.setdefault("last_plan", None)
    st.session_state.setdefault("last_qdf", None)
    st.session_state.setdefault("last_inputs_summary", None)
 
def get_holiday_map(dates):
    if holidays_pkg is None: return {}
    years = sorted(set(pd.to_datetime(dates).year.tolist()))
    us = holidays_pkg.US(years=years)
    out = {}
    for d in dates:
        dd = pd.to_datetime(d).date()
        name = us.get(dd, "")
        if name:
            out[pd.to_datetime(d).normalize()] = str(name)
    return out
 
def build_daily_overrides(plan_dates, weekly_df, closed_dates):
    plan_dates = pd.to_datetime(plan_dates).normalize()
    holiday_map = get_holiday_map(plan_dates)
    w = weekly_df.copy()
    w["Day"] = w["Day"].astype(str)
    w_map = w.set_index("Day")[["Shifts per day", "Hours per shift"]].to_dict(orient="index")
    df = pd.DataFrame({"date": plan_dates})
    df["day"] = df["date"].dt.day_name()
    df["holiday"] = df["date"].map(holiday_map).fillna("")
    df["shifts_per_day"]   = df["day"].apply(lambda d: float(w_map.get(d, {"Shifts per day": 0}).get("Shifts per day", 0)))
    df["hours_per_shift"]  = df["day"].apply(lambda d: float(w_map.get(d, {"Hours per shift": 0.0}).get("Hours per shift", 0.0)))
    closed_set = set(pd.to_datetime(closed_dates).date) if closed_dates else set()
    df["closed"] = df["date"].dt.date.isin(closed_set)
    df["planned_hours"] = df["shifts_per_day"] * df["hours_per_shift"]
    df.loc[df["closed"], ["shifts_per_day", "hours_per_shift", "planned_hours"]] = 0.0
    return df
 
def apply_daily_overrides_to_future(fut, overrides, downtime_min_per_shift):
    out = fut.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    ov = overrides.copy()
    ov["date"] = pd.to_datetime(ov["date"]).dt.normalize()
    out = out.merge(ov[["date", "shifts_per_day", "planned_hours"]], on="date", how="left")
    if "hours_planned_adj" not in out.columns:
        out["hours_planned_adj"] = out["hours_planned"] if "hours_planned" in out.columns else np.nan
    if "hours_down_adj" not in out.columns:
        out["hours_down_adj"] = out["hours_down"] if "hours_down" in out.columns else np.nan
    planned = pd.to_numeric(out["planned_hours"], errors="coerce")
    shifts  = pd.to_numeric(out["shifts_per_day"], errors="coerce")
    mask = planned.notna()
    out.loc[mask, "hours_planned_adj"] = planned.loc[mask].values
    down = (float(downtime_min_per_shift) / 60.0) * shifts
    mask2 = down.notna()
    out.loc[mask2, "hours_down_adj"] = down.loc[mask2].values
    for c in ["chg_hours_adj", "sanitation_hours_adj", "break_hours_adj"]:
        if c not in out.columns: out[c] = 0.0
    out["effective_hours_adj"] = (
        pd.to_numeric(out["hours_planned_adj"], errors="coerce").fillna(0.0)
        - pd.to_numeric(out["hours_down_adj"],   errors="coerce").fillna(0.0)
        - pd.to_numeric(out["chg_hours_adj"],     errors="coerce").fillna(0.0)
        - pd.to_numeric(out["sanitation_hours_adj"], errors="coerce").fillna(0.0)
        - pd.to_numeric(out["break_hours_adj"],   errors="coerce").fillna(0.0)
    )
    out["effective_hours_adj"] = np.maximum(out["effective_hours_adj"], 0.0)
    return out
 
def compute_multipliers_from_user_inputs(cap, weekly_schedule_df, downtime_min_per_shift,
                                          changeovers_per_day, avg_changeover_minutes, eff_percent):
    base_hours = safe_median(cap["hours_planned"]) if "hours_planned" in cap.columns else None
    base_down  = safe_median(cap["hours_down"])    if "hours_down"    in cap.columns else None
    base_chg   = safe_median(cap["chg_hours"])     if "chg_hours"     in cap.columns else None
    _, avg_planned_hours_per_day, avg_shifts_per_day = weekly_schedule_stats(weekly_schedule_df)
    new_hours = float(avg_planned_hours_per_day)
    downtime_hours_per_day   = (float(downtime_min_per_shift) / 60.0) * float(avg_shifts_per_day)
    changeover_hours_per_day = float(changeovers_per_day) * (float(avg_changeover_minutes) / 60.0)
    hours_mult     = safe_ratio(new_hours, base_hours, default=1.0) if base_hours is not None else 1.0
    downtime_mult  = safe_ratio(downtime_hours_per_day,   base_down, default=1.0) if base_down is not None else 1.0
    changeover_mult = safe_ratio(changeover_hours_per_day, base_chg,  default=1.0) if base_chg  is not None else 1.0
    cfr_mult = float(eff_percent) / 100.0
    return hours_mult, downtime_mult, changeover_mult, cfr_mult, {
        "base_hours": base_hours, "base_down": base_down, "base_chg": base_chg,
        "new_hours": new_hours, "new_down": downtime_hours_per_day, "new_chg": changeover_hours_per_day,
        "avg_shifts_per_day": avg_shifts_per_day,
    }
 
def apply_preset_to_session(preset):
    if preset == "Normal week":
        st.session_state.downtime_min_per_shift = 30
        st.session_state.changeovers_per_day = 3.0
        st.session_state.avg_changeover_minutes = 45
        st.session_state.eff_percent = 100.0
    elif preset == "High downtime week":
        st.session_state.downtime_min_per_shift = 75
        st.session_state.changeovers_per_day = 3.0
        st.session_state.avg_changeover_minutes = 45
        st.session_state.eff_percent = 95.0
    elif preset == "Heavy changeover week":
        st.session_state.downtime_min_per_shift = 30
        st.session_state.changeovers_per_day = 6.0
        st.session_state.avg_changeover_minutes = 60
        st.session_state.eff_percent = 98.0
    elif preset == "Staffing shortage":
        st.session_state.downtime_min_per_shift = 45
        st.session_state.changeovers_per_day = 3.0
        st.session_state.avg_changeover_minutes = 45
        st.session_state.eff_percent = 90.0
 
def scenario_count_from_quality(quality):
    return {"Quick (50 futures)": 50, "Standard (200 futures)": 200,
            "Thorough (500 futures)": 500, "Very thorough (1000 futures)": 1000}.get(quality, 200)
 
def costs_from_style(style):
    if style == "Balanced": return 3.0, 1.0
    if style == "Conservative (avoid over-promising)": return 6.0, 0.8
    return 2.0, 1.8
 
def build_one_page_summary(plan, qdf, inputs_summary):
    _bpc  = _plan_gal_col(plan) if plan is not None and not plan.empty else "planned_capacity_units"
    total = float(plan[_bpc].sum()) if plan is not None and not plan.empty else 0.0
    avg   = float(plan[_bpc].mean()) if plan is not None and not plan.empty else 0.0
    risk_note = "N/A"
    try:
        plant_plan = plan.groupby("date", as_index=False)[_plan_gal_col(plan)].sum()
        q50_plant  = qdf.groupby("date", as_index=False)["q50"].sum()
        merged     = plant_plan.merge(q50_plant, on="date", how="left")
        if not merged.empty and "q50" in merged.columns:
            risk_pct  = 100.0 * float((merged["planned_capacity_units"] > merged["q50"]).mean())
            risk_note = f"{risk_pct:.1f}% of days plan exceeds typical (median) capacity"
    except Exception:
        pass
    closed     = inputs_summary.get("closed_dates", [])
    closed_str = ", ".join([str(d) for d in closed]) if closed else "None"
    lines = [
        "PLANT CAPACITY PLAN ONE-PAGE SUMMARY",
        "",
        "Unit note: Q1 line = Quarts (Qt) | G2 line = Gallons (Gal) | P3 line = Pails (5-Gal)",
        "All 'units' totals in this summary add across container types.",
        "",
        "Inputs (what you told the planner):",
        f"- Planning horizon: {inputs_summary.get('horizon_days')} days",
        f"- Uncertainty simulation: {inputs_summary.get('n_scenarios')} futures",
        f"- Preset: {inputs_summary.get('preset')}",
        f"- Plan start date: {inputs_summary.get('plan_start_date')}",
        f"- Plant closed dates: {closed_str}",
        f"- Downtime per shift (min): {inputs_summary.get('downtime_min_per_shift')}",
        f"- Changeovers/day/line: {inputs_summary.get('changeovers_per_day')}",
        f"- Minutes per changeover: {inputs_summary.get('avg_changeover_minutes')}",
        f"- Expected efficiency (CFR %): {inputs_summary.get('eff_percent')}",
        f"- Planning style: {inputs_summary.get('plan_style')}",
        f"- Ramp limit enabled: {inputs_summary.get('ramp_on')} (limit={inputs_summary.get('ramp_limit')})",
        "",
        "Plan results:",
        f"- Total planned output (horizon): {total:,.0f} gal-equiv",
        f"- Average planned per day: {avg:,.0f} gal-equiv/day",
        "- NOTE: All totals are gallon-equivalents (quarts÷4, gallons×1, pails×5).",
        f"- Risk signal: {risk_note}",
        "",
        "Notes:",
        "- Planned hours are derived from your weekly schedule and any daily overrides.",
        "- Holidays can be edited day-by-day in the Daily Overrides table.",
        "- This plan uses historical run rates (CFR) as the baseline; multipliers shift from that baseline.",
        "- Observed data and forecast/modeled outputs are kept separate throughout.",
    ]
    return "\n".join(lines)
 
def plant_chart_with_holidays(merged, holiday_df):
    if alt is None:
        st.line_chart(merged.set_index("date")[["planned_output_units", "typical_units", "low_case_units", "high_case_units"]])
        return
    base  = alt.Chart(merged).encode(x=alt.X("date:T", title="Date"))
    lines = base.transform_fold(
        ["planned_output_units", "typical_units", "low_case_units", "high_case_units"],
        as_=["series", "value"],
    ).mark_line().encode(
        y=alt.Y("value:Q", title="Units"),
        color=alt.Color("series:N", legend=alt.Legend(title="")),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("series:N"), alt.Tooltip("value:Q", format=",.0f")],
    )
    if holiday_df is None or holiday_df.empty:
        st.altair_chart(lines, use_container_width=True)
        return
    rules  = alt.Chart(holiday_df).mark_rule(opacity=0.35).encode(x="date:T")
    labels = alt.Chart(holiday_df).mark_text(align="left", angle=270, dx=3, dy=-5, fontSize=10, opacity=0.7).encode(
        x="date:T", y=alt.value(0), text="label:N")
    st.altair_chart(lines + rules + labels, use_container_width=True)
 
# ─── HEADER ──────────────
cap, q_base, plan_base, plant_base = load_outputs()
 
if cap is None or cap.empty:
    st.error("⚠️ Missing out/capacity_daily.csv. Run the pipeline first: `python 99_run_all.py`")
    st.stop()
 
ensure_session_defaults()
 
# Page header
col_logo, col_title = st.columns([1, 11])
with col_title:
    st.markdown("## 🏭 Plant Capacity Planner")
    st.caption(
        f"Data last refreshed from pipeline outputs · "
        f"Showing plan window: **{st.session_state.plan_start_date}** "
        f"→ **{(pd.Timestamp(st.session_state.plan_start_date) + pd.Timedelta(days=int(st.session_state.horizon_days)-1)).date()}**"
    )
 
start_ts, end_ts = planning_window(st.session_state.plan_start_date, int(st.session_state.horizon_days))
 
# ─── TABS ────────────────
tab_overview, tab_sales, tab_efficiency, tab_whatif = st.tabs([
    "  Capacity Overview",
    "  Sales & Commitments View",
    "  Efficiency Gap Analysis",
    "  What-If Planning Wizard",
])
 
 
# TAB 1 CAPACITY OVERVIEW (Operations Manager)
 
with tab_overview:
    st.markdown('<div class="section-header">Plant Capacity Overview Operations Manager</div>', unsafe_allow_html=True)
 
    # Provenance banner (Step 5.5 framework requirement)
    st.markdown(
        '<div class="provenance-banner">'
        ' <strong>Data source:</strong> Processed pipeline outputs from <code>out/capacity_daily.csv</code> '
        'and <code>out/forecast_capacity_quantiles.csv</code>. '
        'Forecast values are model estimates not observed production. '
        'Historical data shown as solid lines; forecast ranges shown as uncertainty bands. '
        'Update cadence: each time the pipeline is re-run. '
        '<strong>Unit note:</strong> Lines are Barrel Codes (B01, B12, B16, B20, etc.). '
        'Each Barrel can fill Quarts, Gallons, or Pails capacity totals add across all container sizes. '
        'Use the Sales tab for breakdown by container type.'
        '</div>',
        unsafe_allow_html=True,
    )
 
    show_hist = st.checkbox("Include historical data in charts", value=bool(st.session_state.show_history_overview), key="hist_toggle_overview")
    st.session_state.show_history_overview = show_hist
 
    plan_to_show = st.session_state.last_plan if isinstance(st.session_state.last_plan, pd.DataFrame) else plan_base
    qdf_to_show  = st.session_state.last_qdf  if isinstance(st.session_state.last_qdf, pd.DataFrame)  else q_base
 
    if plan_to_show is None or plan_to_show.empty:
        st.info("No planned capacity available yet. Go to the **What-If Planning Wizard** tab and run a plan first.")
    else:
        df = plan_to_show.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        if not show_hist:
            df = filter_to_window(df, start_ts, end_ts)
 
        plant_daily = df.groupby("date", as_index=False)[_plan_gal_col(df)].sum()
        plant_daily = plant_daily.rename(columns={_plan_gal_col(df): "planned_units"})
        total_plan  = float(plant_daily["planned_units"].sum())
 
        # Quantiles
        qdf_plant = None
        if qdf_to_show is not None and not qdf_to_show.empty:
            qtemp = qdf_to_show.copy()
            qtemp["date"] = pd.to_datetime(qtemp["date"]).dt.normalize()
            if not show_hist:
                qtemp = filter_to_window(qtemp, start_ts, end_ts)
            _q10c, _q50c, _q90c = _qdf_gal_cols(qtemp)
            qdf_plant = qtemp.groupby("date", as_index=False).agg(
                q50=("q50","sum"), q10=("q10","sum"), q90=("q90","sum")
            )
 
        # KPI STRIP ───
        if qdf_plant is not None and not qdf_plant.empty:
            merged_kpi = plant_daily.merge(qdf_plant, on="date", how="left")
            total_q50  = float(merged_kpi["q50"].sum())
            total_q10  = float(merged_kpi["q10"].sum())
            total_q90  = float(merged_kpi["q90"].sum())
            stress_label, stress_ratio = plant_stress_indicator(total_plan, total_q50, total_q10)
 
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Total Planned Output (Gal-Equiv)", f"{total_plan:,.0f} gal", help="Gallon-equivalents planned: quarts ÷4, gallons ×1, pails ×5. Comparable across all HSD lines.")
            k2.metric("Typical Achievable (Median)", f"{total_q50:,.0f} gal", help="Median simulated capacity in gallon-equivalents")
            k3.metric("Conservative Estimate (P10)", f"{total_q10:,.0f} gal", help="Low scenario only 10% of simulated futures fall below this")
            k4.metric("Optimistic Estimate (P90)", f"{total_q90:,.0f} gal", help="High scenario 90% of simulated futures fall below this")
            k5.metric(
                "Plan vs. Typical",
                f"{(100.0*(total_plan/total_q50 - 1.0)):+.1f}%" if total_q50 > 0 else "—",
                help="How much your plan exceeds or falls below the typical (median) achievable capacity"
            )
 
            # Plan stress badge
            if stress_ratio is not None:
                badge_class = "status-red" if "High" in stress_label else ("status-yellow" if "Moderate" in stress_label else "status-green")
                st.markdown(
                    f'<span class="{badge_class}">Plant Stress: {stress_label} &nbsp;({stress_ratio:.2f}× plan vs. typical)</span>',
                    unsafe_allow_html=True,
                )
                st.caption("Plan stress compares your planned output to what the plant typically achieves. Values above 1.10× indicate the plan may exceed realistic capacity.")
        else:
            k1, k2, k3, k4 = st.columns(4)
            avg = float(plant_daily["planned_units"].mean())
            peak = float(plant_daily["planned_units"].max())
            low  = float(plant_daily["planned_units"].min())
            k1.metric("Total Planned", f"{total_plan:,.0f}")
            k2.metric("Avg / Day", f"{avg:,.0f}")
            k3.metric("Peak Day", f"{peak:,.0f}")
            k4.metric("Lowest Day", f"{low:,.0f}")
 
        st.divider()
 
        # CAPACITY RISK CHART 
        st.markdown("#### Daily Capacity: Plan vs. Realistic Range")
        st.caption(
            "The **dark band** is your realistic capacity range (P10 to P90 what the plant "
            "achieves 80% of the time). The **plan line** should sit inside this band. "
            "Days where the plan exceeds P90 are flagged as over-committed."
        )

        if qdf_plant is not None and not qdf_plant.empty:
            chart_df = plant_daily.merge(qdf_plant, on="date", how="left").copy()

            # Flag over-committed days (plan > P90)
            chart_df["over_committed"] = chart_df["planned_units"] > chart_df["q90"]
            n_over = int(chart_df["over_committed"].sum())
            n_total = len(chart_df)

            # Risk summary badges
            if n_over == 0:
                st.success(f"✅ Plan is within the realistic range on all {n_total} days.")
            elif n_over <= n_total * 0.1:
                st.warning(f"⚠️ Plan exceeds realistic capacity (P90) on **{n_over} of {n_total} days** review those days.")
            else:
                st.error(f"🔴 Plan exceeds realistic capacity on **{n_over} of {n_total} days** ({n_over/n_total*100:.0f}%) plan may be over-committed.")

            # Build a clean 3-column chart: P10 low, Plan, P90 high
            # Drop P50 it's redundant noise when you have P10/P90
            chart_clean = chart_df.rename(columns={
                "planned_units": "Your Plan (gal)",
                "q90":           "Realistic Max (P90)",
                "q10":           "Realistic Min (P10)",
            })[["date", "Your Plan (gal)", "Realistic Max (P90)", "Realistic Min (P10)"]].set_index("date")

            st.line_chart(chart_clean)
            st.caption(
                "**Realistic Min (P10)** = plant achieves this or better 90% of days.  "
                "**Realistic Max (P90)** = plant achieves this only on its best 10% of days.  "
                "Keep your plan between these two lines."
            )

            # Day-by-day risk table only show if there are issues
            if n_over > 0:
                with st.expander(f"📋 View {n_over} over-committed day(s)"):
                    over_days = chart_df[chart_df["over_committed"]].copy()
                    over_days["Excess (gal)"] = (over_days["planned_units"] - over_days["q90"]).round(0).astype(int)
                    over_days["Plan (gal)"]   = over_days["planned_units"].round(0).astype(int)
                    over_days["P90 Max (gal)"]= over_days["q90"].round(0).astype(int)
                    st.dataframe(
                        over_days[["date", "Plan (gal)", "P90 Max (gal)", "Excess (gal)"]]
                        .rename(columns={"date": "Date"}),
                        use_container_width=True, hide_index=True
                    )
        else:
            # No quantiles just show the plan
            chart_df = plant_daily.copy()
            avg_cap  = float(chart_df["planned_units"].mean())
            chart_df["Avg Capacity"] = avg_cap
            st.line_chart(
                chart_df.rename(columns={"planned_units": "Your Plan (gal)"})
                .set_index("date")[["Your Plan (gal)", "Avg Capacity"]]
            )
            st.caption("Run the What-If Wizard to see the uncertainty band.")

        st.divider()

 
        # WEEKLY OUTPUT BREAKDOWN 
        st.markdown("#### Planned Output by Week Gallons by HSD Line")
        st.caption(
            "Total planned gallon-equivalents per calendar week, broken out by HSD line. "
            "Each colour is one HSD. Use this to spot light or heavy weeks before committing to customers."
        )
        try:
            wb = df.copy()
            wb["date"] = pd.to_datetime(wb["date"]).dt.normalize()
            _wbc = _plan_gal_col(wb)
            wb["week"] = wb["date"].dt.to_period("W-MON").apply(lambda p: p.start_time)
            weekly_by_line = (
                wb.groupby(["week", "line"], as_index=False)[_wbc]
                .sum()
                .rename(columns={_wbc: "gal_equiv"})
            )
            pivot = (
                weekly_by_line
                .pivot(index="week", columns="line", values="gal_equiv")
                .fillna(0.0)
                .sort_index()
            )
            pivot.index = pivot.index.strftime("Wk %b %d")
            weekly_total = pivot.sum(axis=1)
            wt1, wt2, wt3 = st.columns(3)
            wt1.metric("Avg Weekly Output (gal)", f"{float(weekly_total.mean()):,.0f}",
                       help="Average planned gal-equiv per week")
            wt2.metric("Lightest Week (gal)",     f"{float(weekly_total.min()):,.0f}",
                       help="Lowest planned week may indicate shutdowns or light scheduling")
            wt3.metric("Heaviest Week (gal)",     f"{float(weekly_total.max()):,.0f}",
                       help="Peak planned week")
            st.bar_chart(pivot)
            st.caption(
                "Each colour = one HSD line. All values in gallon-equivalents "
                "(quarts ÷4 · gallons ×1 · pails ×5)."
            )
        except Exception as e:
            st.info(f"Weekly breakdown unavailable: {e}")

        st.divider()

 
        # SELLABLE CAPACITY TABLE (Operations Manager) ──────────────────────
        st.markdown("#### Planned Output by Line Next 7 and 30 Days")
        st.caption(
            "Shows total planned units per production line for the next 7 and 30 days. "
            "**Q1 = Quarts · G2 = Gallons · P3 = Pails (5-gal).** "
            "'Plan vs. Typical' compares your plan against the median simulated capacity for that line "
            "values above 1.0× mean the plan exceeds typical achievable output."
        )
        today_dt = df["date"].min()
        end7  = today_dt + pd.Timedelta(days=7)
        end30 = today_dt + pd.Timedelta(days=30)
        df7   = df[(df["date"] >= today_dt) & (df["date"] < end7)]
        df30  = df[(df["date"] >= today_dt) & (df["date"] < end30)]
        _gc_line = _plan_gal_col(df) if not df.empty else "planned_capacity_units"
        by7   = df7.groupby("line",  as_index=False)[_gc_line].sum().rename(columns={_gc_line: "Next 7 Days (gal)"})
        by30  = df30.groupby("line", as_index=False)[_gc_line].sum().rename(columns={_gc_line: "Next 30 Days (gal)"})
        tbl   = by7.merge(by30, on="line", how="outer").fillna(0.0)
 
        if qdf_to_show is not None and isinstance(qdf_to_show, pd.DataFrame) and "q50" in qdf_to_show.columns and "line" in qdf_to_show.columns:
            qtemp30 = qdf_to_show.copy()
            qtemp30["date"] = pd.to_datetime(qtemp30["date"]).dt.normalize()
            q30     = qtemp30[(qtemp30["date"] >= today_dt) & (qtemp30["date"] < end30)]
            qline   = q30.groupby("line", as_index=False)["q50"].sum().rename(columns={"q50": "Typical (30 Days, gal)"})
            tbl     = tbl.merge(qline, on="line", how="left")
            tbl["Plan vs. Typical"] = tbl.apply(
                lambda r: _safe_div(float(r["Next 30 Days (gal)"]), float(r["Typical (30 Days, gal)"])) if pd.notna(r.get("Typical (30 Days, gal)", np.nan)) else np.nan, axis=1)
            tbl["Status"] = tbl["Plan vs. Typical"].map(_risk_flag_from_ratio)
 
        show = tbl.copy()
        for c in ["Next 7 Days (gal)", "Next 30 Days (gal)", "Typical (30 Days, gal)"]:
            if c in show.columns:
                show[c] = show[c].map(lambda x: f"{float(x):,.0f} gal" if pd.notna(x) else "—")
        if "Plan vs. Typical" in show.columns:
            show["Plan vs. Typical"] = show["Plan vs. Typical"].map(lambda x: f"{x:.2f}×" if pd.notna(x) else "—")
 
        rename_line = {"line": "Production Line"}
        show = show.rename(columns=rename_line)
        # HSD lines can fill multiple container sizes show color family instead
        hsd_color_map = {
            "HM01": "White (6A)", "HM36": "White (6A)", "HM56": "White (6A)",
            "HM20": "Black (5A)", "HM25": "Black (5A)", "HM30": "Black (5A)",
            "HM12": "Blue/Green (7A)", "HM13": "Blue/Green (7A)", "HM14": "Blue/Green (7A)",
            "HM16": "Red (7B)", "HM43": "Red (7B)", "HM47": "Red (7B)",
            "HM50": "Clear (8A)",
        }
        show.insert(1, "Color / W/C", show["Production Line"].map(
            lambda l: hsd_color_map.get(str(l), str(l))))
        st.dataframe(show.sort_values("Production Line"), use_container_width=True, hide_index=True)
 
        st.divider()
 
        # LINE-BY-LINE EXPANDER 
        with st.expander("📈 Line-by-line daily charts (expand to view)"):
            for ln in sorted(df["line"].dropna().unique()):
                st.markdown(f"**Line: {ln}**")
                _lc2 = _plan_gal_col(df)
                s = df[df["line"] == ln].sort_values("date")[["date", _lc2]]
                s = s.rename(columns={_lc2: "Planned Gallons (Gal-Equiv)"})
                st.line_chart(s.set_index("date"))
 
    st.divider()
 
    # SKU SPEED RANKING 
    st.markdown('<div class="section-header">🏎️ SKU Run Speed Ranking</div>', unsafe_allow_html=True)
    st.caption(
        "Ranks SKUs from fastest to slowest based on median units produced per hour (CFR units/hour). "
        "Faster SKUs generate more output per shift. "
        "Use this table when discussing what product mix affects capacity."
    )
    try:
        sku_ranks = pd.read_csv("out/sku_ranks.csv").sort_values("rank_fastest_first")
        sku_ranks["container_display"] = sku_ranks["container_type"].apply(container_display_name)

        # Build display columns include color_family and wc_grade if present
        display_cols = ["rank_fastest_first", "sku", "container_display", "median_cfr_uh"]
        rename_map = {
            "rank_fastest_first": "Speed Rank",
            "sku":                "SKU (Product-Color-Container)",
            "container_display":  "Container Type",
            "median_cfr_uh":      "Median Units / Hour",
        }
        if "color_family" in sku_ranks.columns:
            display_cols.append("color_family")
            rename_map["color_family"] = "Color Family"
        if "wc_grade" in sku_ranks.columns:
            display_cols.append("wc_grade")
            rename_map["wc_grade"] = "W/C Grade"
        if "lines_seen" in sku_ranks.columns:
            display_cols.append("lines_seen")
            rename_map["lines_seen"] = "HSDs Run On"

        st.dataframe(
            sku_ranks[display_cols].rename(columns=rename_map),
            use_container_width=True, hide_index=True,
        )
    except Exception:
        st.info("SKU speed ranking not available sku_ranks.csv not found in output directory.")
 
 
 
# TAB 2 SALES & COMMITMENTS VIEW
 
with tab_sales:
    st.markdown('<div class="section-header"> Sales & Commitments View What Can We Promise?</div>', unsafe_allow_html=True)
 
    # Provenance banner
    st.markdown(
        '<div class="provenance-banner">'
        '<strong>What this view shows:</strong> Total available capacity over the planning horizon, '
        'broken down by container type: <strong>Quarts (Qt)</strong>, <strong>Gallons (Gal)</strong>, '
        'and <strong>Pails (5-Gal)</strong>. These numbers are <em>model-derived estimates</em> based on historical '
        'run rates not confirmed production orders. Always validate against the latest run plan before '
        'making customer commitments. <strong>Forecast uncertainty band applies.</strong>'
        '</div>',
        unsafe_allow_html=True,
    )
 
    plan_for_sales = st.session_state.last_plan if isinstance(st.session_state.last_plan, pd.DataFrame) else plan_base
    qdf_for_sales  = st.session_state.last_qdf  if isinstance(st.session_state.last_qdf, pd.DataFrame)  else q_base
 
    if plan_for_sales is None or plan_for_sales.empty:
        st.info("No plan available yet. Go to the **What-If Planning Wizard** tab, configure your assumptions, and click **Run Plan**.")
    else:
        pp = plan_for_sales.copy()
        pp["date"] = pd.to_datetime(pp["date"]).dt.normalize()
        pp_window   = filter_to_window(pp, start_ts, end_ts)
        _pgc        = _plan_gal_col(pp_window)
        total_units = float(pp_window[_pgc].sum())        # gallon-equivalents
        avg_daily   = float(pp_window[_pgc].mean()) if not pp_window.empty else 0.0
 
        # TOP KPIs ─────
        s1, s2, s3 = st.columns(3)
        s1.metric(
            f"Total Available Capacity ({int(st.session_state.horizon_days)}-Day Window)",
            f"{total_units:,.0f} gallons",
            help="Total planned gallon-equivalents across all HSD lines (quarts÷4 + gallons×1 + pails×5)"
        )
        s2.metric(
            "Average Daily Output",
            f"{avg_daily:,.0f} gal / day",
            help="Average planned gallon-equivalents per day"
        )
        # 7-day forward
        pp_sorted   = pp_window.groupby("date", as_index=False)[_pgc].sum().sort_values("date")
        pp_sorted["fwd_7d"] = pp_sorted[_pgc].rolling(window=7, min_periods=7).sum().shift(-6)
        fwd_7 = pp_sorted.dropna(subset=["fwd_7d"])
        s3.metric(
            "Next 7-Day Forecast",
            f"{float(fwd_7.iloc[0]['fwd_7d']):,.0f} units" if not fwd_7.empty else "—",
            help="Total planned output for the next 7 days from the start of the planning window"
        )
 
        # CONTAINER MIX 
        st.divider()
        st.markdown("#### Estimated Capacity by Container Type (Quarts · Gallons · Pails)")
        st.caption(
            "Capacity is allocated across container types **Quart (Qt)**, **Gallon (Gal)**, and **Pail (5-Gal)** "
            "proportionally based on each line's historical run speed (median units/hour). "
            "**This is an assumed mix not a scheduled SKU plan.** "
            "It helps sales understand what product types the plant is most efficient at producing."
        )
 
        try:
            sku_ranks_df = pd.read_csv("out/sku_ranks.csv")
            container_mix = allocate_capacity_by_container_from_sku_ranks(
                total_units=total_units, sku_ranks_df=sku_ranks_df, use_speed_col="median_cfr_uh"
            )
            if container_mix.empty:
                st.info("Container mix not available check sku_ranks.csv.")
            else:
                # Metric cards per container
                cols = st.columns(min(4, len(container_mix)))
                for i, (_, r) in enumerate(container_mix.head(4).iterrows()):
                    cols[i].metric(
                        container_display_name(str(r['container_type'])),
                        f"{float(r['planned_units']):,.0f} units",
                        f"{float(r['share_pct']):.1f}% of total",
                    )
 
                # Full breakdown table
                mix_display = container_mix.copy()
                mix_display["container_type"] = mix_display["container_type"].apply(container_display_name)
                mix_display["Planned Units"] = mix_display["planned_units"].map(lambda x: f"{x:,.0f}")
                mix_display["Share of Total"] = mix_display["share_pct"].map(lambda x: f"{x:.1f}%")
                mix_display = mix_display.rename(columns={"container_type": "Container Type"})
                st.dataframe(mix_display[["Container Type", "Planned Units", "Share of Total"]], use_container_width=True, hide_index=True)
 
            # MIX RISK SIGNAL 
            slow_share, slow_flag = slow_sku_dominance_flag(sku_ranks_df)
            st.divider()
            st.markdown("#### Product Mix Risk Signal")
            st.caption(
                "If the product mix contains many slow-running SKUs, total output capacity will be lower "
                "than the headline number suggests. 🔴 means slow SKUs make up a large share of the assumed mix."
            )
            badge_class = "status-red" if "🔴" in slow_flag else ("status-yellow" if "🟡" in slow_flag else "status-green")
            st.markdown(
                f'<span class="{badge_class}">{slow_flag}</span> '
                f'&nbsp; Slow SKU group share of assumed mix: <strong>{slow_share:.1f}%</strong>',
                unsafe_allow_html=True,
            )
 
        except Exception as e:
            st.info("Container mix requires sku_ranks.csv in the app working directory.")
            st.caption(str(e))
 
        # UNCERTAINTY REMINDER 
        st.divider()
        if qdf_for_sales is not None and not qdf_for_sales.empty:
            qtemp = qdf_for_sales.copy()
            qtemp["date"] = pd.to_datetime(qtemp["date"]).dt.normalize()
            qtemp = filter_to_window(qtemp, start_ts, end_ts)
            q50_total = float(qtemp["q50"].sum()) if "q50" in qtemp.columns else None
            q10_total = float(qtemp["q10"].sum()) if "q10" in qtemp.columns else None
            q90_total = float(qtemp["q90"].sum()) if "q90" in qtemp.columns else None
 
            st.markdown("#### Capacity Uncertainty Range")
            st.caption(
                "These ranges come from simulating many possible futures using historical variability. "
                "**For conservative commitments, use the P10 (low) figure.** "
                "P50 is the most likely outcome. P90 is achievable only in a strong week."
            )
            u1, u2, u3 = st.columns(3)
            u1.metric("Conservative Commitment (P10)", f"{q10_total:,.0f} gal" if q10_total else "—",
                      help="Only 10% of simulated futures fall below this safe for firm commitments")
            u2.metric("Most Likely Capacity (P50)", f"{q50_total:,.0f} gal" if q50_total else "—",
                      help="The median simulated capacity half of futures are above, half below")
            u3.metric("Best-Case Capacity (P90)", f"{q90_total:,.0f} gal" if q90_total else "—",
                      help="Only achievable in 10% of simulated futures do not commit at this level")
 
            st.info(
                "💡 **For Sales:** Use the **Conservative (P10)** figure for firm customer commitments. "
                "Use **Most Likely (P50)** for internal planning. "
                "Never commit based on the **Best-Case (P90)** figure."
            )
 
 
 
# TAB 3 WHAT-IF PLANNING WIZARD
 
with tab_whatif:
    st.markdown('<div class="section-header">⚙️ What-If Planning Wizard Build a Capacity Plan</div>', unsafe_allow_html=True)
 
    st.markdown(
        """
        Use this wizard to model a capacity plan for the upcoming period.
        Work through each step, adjust assumptions to match your expected week, and click **Run Plan** at the bottom.
        Your results will appear here and update the **Capacity Overview** and **Sales View** tabs automatically.
        """
    )
 
    st.info("ℹ️ Complete all steps below, then click **Run Plan** to generate your capacity forecast.")
 
    # STEP 1: Planning Window 
    st.markdown("---")
    st.markdown("### <span class='step-pill'>1</span> How long are you planning for?", unsafe_allow_html=True)
    st.caption("Choose your planning start date and how many days into the future to model.")
 
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state.plan_start_date = st.date_input(
            "Plan start date",
            value=st.session_state.plan_start_date,
            help="The plan will begin on this date and run forward for the number of days you select.",
        )
    with c2:
        st.session_state.horizon_days = st.number_input(
            "Number of days to plan",
            min_value=7, max_value=90,
            value=int(st.session_state.horizon_days), step=1,
            help="Typically 7–30 days for short-term operations, up to 90 days for medium-range planning.",
        )
    with c3:
        options_quality = ["Quick (50 futures)", "Standard (200 futures)", "Thorough (500 futures)", "Very thorough (1000 futures)"]
        st.session_state.scenario_quality = st.selectbox(
            "Simulation quality",
            options=options_quality,
            index=options_quality.index(st.session_state.scenario_quality) if st.session_state.scenario_quality in options_quality else 1,
            help="More futures = more accurate uncertainty bands, but slower to run. Standard is fine for most use cases.",
        )
 
    start_ts, end_ts = planning_window(st.session_state.plan_start_date, int(st.session_state.horizon_days))
    st.caption(f"📅 Planning window: **{start_ts.date()}** through **{end_ts.date()}** ({int(st.session_state.horizon_days)} days)")
 
    # STEP 2: Operational Preset 
    st.markdown("---")
    st.markdown("### <span class='step-pill'>2</span> What kind of week is this?", unsafe_allow_html=True)
    st.caption(
        "Choose a preset that best matches your expected operating conditions. "
        "This sets defaults for downtime, changeovers, and efficiency you can fine-tune in Step 4."
    )
 
    preset_options = ["Normal week", "High downtime week", "Heavy changeover week", "Staffing shortage", "Custom"]
    preset_descriptions = {
        "Normal week": "Typical operations standard downtime, normal changeover schedule, full efficiency.",
        "High downtime week": "More equipment issues or scheduled maintenance than usual.",
        "Heavy changeover week": "Many product/SKU changeovers expected reduces net run time.",
        "Staffing shortage": "Reduced crew expect lower throughput and efficiency.",
        "Custom": "Set all values manually in Step 4 below.",
    }
    preset = st.selectbox(
        "Choose a scenario preset",
        options=preset_options,
        index=preset_options.index(st.session_state.preset) if st.session_state.preset in preset_options else 0,
        help="Presets load typical values for that scenario type. You can override any value in Step 4.",
    )
    if preset != st.session_state.preset:
        st.session_state.preset = preset
        apply_preset_to_session(preset)
 
    st.caption(f"_{preset_descriptions.get(preset, '')}_")
 
    # STEP 3: Schedule & Holidays 
    st.markdown("---")
    st.markdown("### <span class='step-pill'>3</span> Set your shift schedule and mark any closures", unsafe_allow_html=True)
    st.caption(
        "Define how many shifts are running each day of the week and how long each shift is. "
        "Then review the daily calendar below check the 'Closed' box for any holidays or shutdown days."
    )
 
    # Weekly schedule editor
    st.markdown("**Baseline weekly schedule** hours planned per day = shifts × hours per shift")
    weekly_schedule = st.data_editor(
        st.session_state.weekly_schedule_df,
        use_container_width=True, hide_index=True,
        column_config={
            "Day": st.column_config.TextColumn(disabled=True),
            "Shifts per day": st.column_config.NumberColumn(min_value=0, max_value=4, step=1, help="Number of production shifts this day"),
            "Hours per shift": st.column_config.NumberColumn(min_value=0.0, max_value=16.0, step=0.5, help="Length of each shift in hours"),
        },
        key="weekly_schedule_editor",
    )
    st.session_state.weekly_schedule_df = weekly_schedule
 
    # Daily overrides calendar
    plan_dates = pd.date_range(start_ts, end_ts, freq="D")
    base_override = build_daily_overrides(plan_dates, weekly_schedule, st.session_state.closed_dates)
 
    def _override_dates_sig(df):
        if df is None or df.empty or "date" not in df.columns: return tuple()
        return tuple(pd.to_datetime(df["date"]).dt.normalize().tolist())
 
    if (st.session_state.daily_override_df is None or
            _override_dates_sig(st.session_state.daily_override_df) != _override_dates_sig(base_override)):
        st.session_state.daily_override_df = base_override
 
    st.markdown("**Daily calendar mark closures and adjust any day individually**")
    st.caption(
        "✅ Check **Closed** to set a day to zero production (holidays, shutdowns). "
        "You can also override shifts or hours for a specific day without closing it."
    )
 
    edited = st.data_editor(
        st.session_state.daily_override_df,
        use_container_width=True, hide_index=True,
        column_config={
            "date":           st.column_config.DateColumn(disabled=True, label="Date"),
            "day":            st.column_config.TextColumn(disabled=True, label="Day of Week"),
            "holiday":        st.column_config.TextColumn(disabled=True, label="Holiday"),
            "closed":         st.column_config.CheckboxColumn(label="Closed (zero production)"),
            "shifts_per_day": st.column_config.NumberColumn(min_value=0, max_value=4, step=1, label="Shifts"),
            "hours_per_shift":st.column_config.NumberColumn(min_value=0.0, max_value=16.0, step=0.5, label="Hrs/Shift"),
            "planned_hours":  st.column_config.NumberColumn(disabled=True, label="Total Planned Hours"),
        },
        key="daily_override_editor",
    )
 
    edited = edited.copy()
    edited["date"] = pd.to_datetime(edited["date"]).dt.normalize()
    edited["shifts_per_day"]  = pd.to_numeric(edited["shifts_per_day"],  errors="coerce").fillna(0.0)
    edited["hours_per_shift"] = pd.to_numeric(edited["hours_per_shift"], errors="coerce").fillna(0.0)
    edited["closed"]          = edited["closed"].astype(bool)
    edited["planned_hours"]   = edited["shifts_per_day"] * edited["hours_per_shift"]
    edited.loc[edited["closed"], ["shifts_per_day", "hours_per_shift", "planned_hours"]] = 0.0
    st.session_state.daily_override_df = edited
    st.session_state.closed_dates = sorted(list(pd.to_datetime(edited.loc[edited["closed"], "date"]).dt.date.unique()))
 
    if st.session_state.closed_dates:
        st.caption(f"🔴 Closed dates in this horizon: **{', '.join([str(d) for d in st.session_state.closed_dates])}**")
 
    if holidays_pkg is None:
        st.caption("💡 Tip: Install `pip install holidays` to auto-label US federal holidays in the calendar.")
    else:
        if st.button("📅 Auto-mark all federal holidays in this window as CLOSED"):
            hol_map   = get_holiday_map(plan_dates)
            hol_dates = sorted([d.date() for d in hol_map.keys()])
            st.session_state.closed_dates = sorted(list(set(st.session_state.closed_dates).union(set(hol_dates))))
            st.session_state.daily_override_df = None
            st.rerun()
 
    # STEP 4: Operational Assumptions ───────────────────────────────────────
    st.markdown("---")
    st.markdown("### <span class='step-pill'>4</span> Fine-tune your operational assumptions", unsafe_allow_html=True)
    st.caption(
        "These inputs adjust how the model scales from historical typical performance. "
        "Leave at preset defaults unless you have specific information about the upcoming period."
    )
 
    c3, c4, c5 = st.columns(3)
    with c3:
        st.session_state.downtime_min_per_shift = st.number_input(
            "Expected downtime per shift (minutes)",
            min_value=0, max_value=240,
            value=int(st.session_state.downtime_min_per_shift), step=5,
            help="Average unplanned downtime per shift: breakdowns, minor stoppages, waiting. Typical baseline is ~30 min.",
        )
    with c4:
        st.session_state.eff_percent = st.number_input(
            "Expected line efficiency (%)",
            min_value=50.0, max_value=120.0,
            value=float(st.session_state.eff_percent), step=1.0,
            help="100% = historical typical performance. 90% = slower than usual. 110% = unusually strong run.",
        )
    with c5:
        st.session_state.changeovers_per_day = st.number_input(
            "Changeovers per line per day (avg)",
            min_value=0.0, max_value=20.0,
            value=float(st.session_state.changeovers_per_day), step=0.5,
            help="Each changeover takes the line offline for setup. More changeovers = less run time.",
        )
        st.session_state.avg_changeover_minutes = st.number_input(
            "Minutes per changeover (avg)",
            min_value=0, max_value=240,
            value=int(st.session_state.avg_changeover_minutes), step=5,
            help="Average time to complete one product/SKU changeover.",
        )
 
    # Warnings based on inputs
    if st.session_state.downtime_min_per_shift > 120:
        st.warning("⚠️ Downtime is set above 120 minutes per shift this is very high and will significantly reduce planned capacity.")
    if st.session_state.changeovers_per_day >= 10:
        st.warning("⚠️ More than 10 changeovers per line per day is unusually high expect a large reduction in run time.")
    if st.session_state.eff_percent < 80:
        st.warning("⚠️ Efficiency below 80% will produce a very conservative plan. Confirm this reflects real expected conditions.")
 
    # STEP 5: Planning Style ─
    st.markdown("---")
    st.markdown("### <span class='step-pill'>5</span> How should the plan handle uncertainty?", unsafe_allow_html=True)
    st.caption(
        "The model simulates many possible future scenarios based on historical variability. "
        "Your planning style controls how cautious or aggressive the final plan target is."
    )
 
    style_options = ["Balanced", "Conservative (avoid over-promising)", "Aggressive (maximize output)"]
    style_descriptions = {
        "Balanced": "Equal weight on shortfalls and overages. Good default for most planning.",
        "Conservative (avoid over-promising)": "Penalizes overages more generates a plan less likely to miss commitments. Recommended for customer-facing commitments.",
        "Aggressive (maximize output)": "Pushes for higher output targets. Higher risk of falling short.",
    }
    st.session_state.plan_style = st.radio(
        "Planning style",
        options=style_options,
        index=style_options.index(st.session_state.plan_style) if st.session_state.plan_style in style_options else 0,
        help="Affects the cost trade-off in the stochastic optimization not just a label.",
    )
    st.caption(f"_{style_descriptions.get(st.session_state.plan_style, '')}_")
 
    with st.expander("🔧 Advanced options (optional leave as defaults unless you have a specific reason to change)"):
        st.session_state.ramp_on = st.checkbox(
            "Limit day-to-day output swings",
            value=bool(st.session_state.ramp_on),
            help="Prevents the plan from jumping sharply between days. Useful for smooth scheduling.",
        )
        st.session_state.ramp_limit = st.number_input(
            "Maximum daily change per line (units/day)",
            min_value=0.0, value=float(st.session_state.ramp_limit), step=500.0,
        )
        st.session_state.show_holiday_markers = st.checkbox(
            "Show holiday markers on charts",
            value=bool(st.session_state.show_holiday_markers),
        )
 
    # PREVIEW ──────────
    st.markdown("---")
    st.markdown("### Preview: How your inputs translate to the model")
    n_scenarios = scenario_count_from_quality(st.session_state.scenario_quality)
 
    hours_mult, downtime_mult, changeover_mult, cfr_mult, debug_info = compute_multipliers_from_user_inputs(
        cap,
        weekly_schedule_df=weekly_schedule,
        downtime_min_per_shift=int(st.session_state.downtime_min_per_shift),
        changeovers_per_day=float(st.session_state.changeovers_per_day),
        avg_changeover_minutes=int(st.session_state.avg_changeover_minutes),
        eff_percent=float(st.session_state.eff_percent),
    )
 
    st.caption(
        "The model works by comparing your inputs to the historical typical values from the pipeline. "
        "Multipliers below show how much each factor is shifted relative to the historical baseline. "
        "1.0× means 'same as typical'; 0.9× means 10% less than typical."
    )
 
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Hours Multiplier",     f"{hours_mult:.2f}×",     help="Your planned hours vs historical median planned hours")
    p2.metric("Downtime Multiplier",  f"{downtime_mult:.2f}×",  help="Your expected downtime vs historical median downtime")
    p3.metric("Changeover Multiplier",f"{changeover_mult:.2f}×",help="Your changeover load vs historical median changeover hours")
    p4.metric("Efficiency Multiplier",f"{cfr_mult:.2f}×",       help="Your efficiency % ÷ 100")
 
    st.caption(
        f"Simulation will run **{n_scenarios} futures** · "
        f"Planning style: **{st.session_state.plan_style}** · "
        f"Window: **{start_ts.date()} → {end_ts.date()}**"
    )
 
    # RUN PLAN ─────────
    st.markdown("---")
    st.markdown("### <span class='step-pill'>6</span> Run the plan", unsafe_allow_html=True)
    st.caption("Click the button below to generate your capacity plan. This may take a few seconds depending on simulation quality.")
 
    run = st.button("▶ Run Plan", type="primary", use_container_width=True)
 
    if run:
        cost_under, cost_over = costs_from_style(st.session_state.plan_style)
        ramp = float(st.session_state.ramp_limit) if st.session_state.ramp_on else None
 
        with st.spinner("Running capacity simulation..."):
            hist_adj = apply_what_if_to_drivers(
                cap,
                hours_mult=hours_mult, downtime_mult=downtime_mult,
                changeover_mult=changeover_mult, cfr_mult=cfr_mult,
            )
 
            driver_cols_adj = [
                "hours_planned_adj", "hours_down_adj", "chg_hours_adj",
                "sanitation_hours_adj", "break_hours_adj", "effective_hours_adj",
                "cfr_avg_uh_adj", "cfr_p10_uh_adj", "cfr_p90_uh_adj",
                "cfr_avg_gph_adj", "cfr_p10_gph_adj", "cfr_p90_gph_adj",  # gallon-equiv CFR
            ]
 
            try:
                fut = make_future_driver_rows_from_history(
                    hist_adj, horizon_days=int(st.session_state.horizon_days),
                    driver_cols=driver_cols_adj, start_date=start_ts,
                )
            except TypeError:
                fut = make_future_driver_rows_from_history(hist_adj, int(st.session_state.horizon_days), driver_cols_adj)
                fut = fut.copy()
                fut["date"] = pd.date_range(start_ts, end_ts, freq="D").repeat(
                    len(fut["line"].dropna().unique()) if "line" in fut.columns else 1
                ).values[:len(fut)]
 
            fut  = apply_daily_overrides_to_future(fut, edited, int(st.session_state.downtime_min_per_shift))
            qdf  = quantile_band_from_adjusted_drivers(fut)
            scenarios = sample_scenarios_from_quantiles(qdf, n_scenarios=int(n_scenarios), seed=42)
            plan = solve_stochastic_plan_ortools(scenarios, cost_under=cost_under, cost_over=cost_over, ramp_limit=ramp)
 
        plan["date"] = pd.to_datetime(plan["date"]).dt.normalize()
        qdf["date"]  = pd.to_datetime(qdf["date"]).dt.normalize()
        plan = filter_to_window(plan, start_ts, end_ts)
        qdf  = filter_to_window(qdf,  start_ts, end_ts)
 
        st.session_state.last_plan = plan
        st.session_state.last_qdf  = qdf
        st.session_state.last_inputs_summary = {
            "horizon_days": int(st.session_state.horizon_days),
            "n_scenarios": int(n_scenarios),
            "preset": st.session_state.preset,
            "plan_start_date": str(st.session_state.plan_start_date),
            "closed_dates": [str(d) for d in st.session_state.closed_dates],
            "downtime_min_per_shift": int(st.session_state.downtime_min_per_shift),
            "changeovers_per_day": float(st.session_state.changeovers_per_day),
            "avg_changeover_minutes": int(st.session_state.avg_changeover_minutes),
            "eff_percent": float(st.session_state.eff_percent),
            "plan_style": st.session_state.plan_style,
            "ramp_on": bool(st.session_state.ramp_on),
            "ramp_limit": float(st.session_state.ramp_limit),
            "internal_multipliers": {"hours_mult": float(hours_mult), "downtime_mult": float(downtime_mult),
                                      "changeover_mult": float(changeover_mult), "cfr_mult": float(cfr_mult)},
            "internal_costs": {"cost_under": float(cost_under), "cost_over": float(cost_over)},
        }
 
        st.success(
            f"✅ Plan generated successfully for **{start_ts.date()} → {end_ts.date()}**. "
            f"Switch to the **Capacity Overview** or **Sales & Commitments View** tabs to review results."
        )
 
        # RESULTS SUMMARY 
        _wic = _plan_gal_col(plan)
        plant_plan = plan.groupby("date", as_index=False)[_wic].sum()
        plant_plan = plant_plan.rename(columns={_wic: "planned_output_units"})
        q50_plant  = qdf.groupby("date", as_index=False)["q50"].sum().rename(columns={"q50": "typical_units"})
        q10_plant  = qdf.groupby("date", as_index=False)["q10"].sum().rename(columns={"q10": "conservative_units"})
        q90_plant  = qdf.groupby("date", as_index=False)["q90"].sum().rename(columns={"q90": "optimistic_units"})
        merged = (plant_plan.merge(q50_plant, on="date", how="left")
                             .merge(q10_plant, on="date", how="left")
                             .merge(q90_plant, on="date", how="left"))
 
        total = float(merged["planned_output_units"].sum())
        avg   = float(merged["planned_output_units"].mean())
        peak  = float(merged["planned_output_units"].max())
        low   = float(merged["planned_output_units"].min())
 
        st.markdown("#### Plan Results Summary")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Total Planned Output", f"{total:,.0f} gal")
        r2.metric("Average per Day",      f"{avg:,.0f} gal/day")
        r3.metric("Peak Day",             f"{peak:,.0f} gal")
        r4.metric("Lowest Day",           f"{low:,.0f} gal")
 
        try:
            risk_pct = 100.0 * float((merged["planned_output_units"] > merged["typical_units"]).mean())
            if risk_pct > 30:
                st.warning(f"⚠️ On {risk_pct:.1f}% of days, your plan exceeds typical (median) achievable capacity. Consider switching to Conservative style.")
            else:
                st.caption(f"ℹ️ On {risk_pct:.1f}% of days, your plan exceeds typical achievable capacity within a reasonable range.")
        except Exception:
            pass
 
        st.markdown("#### Planned Output vs. Capacity Scenarios")
        st.caption(
            "**Solid line** = your planned output commitment. "
            "**Dashed/band lines** = model estimates of what the plant can achieve (not observed data). "
            "The band shows the range from conservative (P10) to optimistic (P90)."
        )
 
        holiday_markers = None
        if bool(st.session_state.show_holiday_markers):
            hol = edited[edited["holiday"].astype(str).str.len() > 0][["date", "holiday"]].drop_duplicates().copy()
            if not hol.empty:
                hol["label"] = hol["holiday"].astype(str)
                holiday_markers = hol[["date", "label"]]
 
        chart_merged = merged.rename(columns={
            "planned_output_units": "Your Plan",
            "typical_units": "Typical Achievable (P50)",
            "conservative_units": "Conservative (P10)",
            "optimistic_units": "Optimistic (P90)",
        })
        if holiday_markers is not None and alt is not None:
            plant_chart_with_holidays(merged, holiday_markers)
        else:
            st.line_chart(chart_merged.set_index("date")[["Your Plan", "Typical Achievable (P50)", "Conservative (P10)", "Optimistic (P90)"]])
 
        with st.expander("📋 View daily detail table"):
            st.dataframe(chart_merged.rename(columns={"date": "Date"}), use_container_width=True, hide_index=True)
 
        with st.expander("📈 Line-by-line output (expand)"):
            for ln in sorted(plan["line"].dropna().unique()):
                st.markdown(f"**Line {ln}**")
                _lc = _plan_gal_col(plan)
                s = plan[plan["line"] == ln].sort_values("date")[["date", _lc]]
                st.line_chart(s.rename(columns={_lc: "Planned Gallons (Gal-Equiv)"}).set_index("date"))
 
        # DOWNLOADS 
        st.markdown("#### Download Results")
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "⬇️ Download detailed plan (CSV by line & day)",
                data=plan.to_csv(index=False).encode("utf-8"),
                file_name="planned_capacity_daily.csv",
                mime="text/csv",
            )
        with d2:
            summary_txt = build_one_page_summary(plan, qdf, st.session_state.last_inputs_summary)
            st.download_button(
                "⬇️ Download one-page plan summary (TXT)",
                data=summary_txt.encode("utf-8"),
                file_name="plan_summary.txt",
                mime="text/plain",
            )
 
 
# TAB 3 EFFICIENCY GAP ANALYSIS
 
with tab_efficiency:
    st.markdown('<div class="section-header">Efficiency Gap Analysis Where Is Capacity Being Lost?</div>', unsafe_allow_html=True)
 
    st.markdown(
        """
        This tab answers the question operations managers and plant leaders actually need:
        **"We know we're not running at 100% but what exactly are we leaving on the table,
        and what does getting to 90% efficiency look like in real units?"**
 
        The analysis uses your actual historical run data as the baseline.
        The "ceiling" is your own plant's P90 CFR what your best days actually look like,
        not a theoretical maximum. This keeps the targets grounded in reality.
        """
    )
 
    st.markdown(
        '<div class="provenance-banner">'
        '<strong>Data source:</strong> <code>out/capacity_daily.csv</code> historical realized production. '
        'All values are derived from observed run data. '
        'The "best-day ceiling" is your plant\'s own 90th-percentile CFR (units/hour), '
        'not an engineering theoretical maximum. '
        'Forecast scenarios are model estimates and shown separately.'
        '</div>',
        unsafe_allow_html=True,
    )
 
    if cap is None or cap.empty:
        st.info("Capacity data not available. Run the pipeline first.")
    else:
        try:
            gap_data = compute_efficiency_gap(cap)
            trend_data = compute_cfr_trend(
                cap,
                window_days=30,
                ceiling_p90=gap_data.get("p90_cfr"),   # use same ceiling as KPI badge
            )
 
            # Top KPI strip ─
            current_eff = gap_data["current_eff_pct"]
            gap_units   = gap_data["gap_units_90pct"]
            wf          = gap_data["loss_waterfall"]
 
            # Efficiency badge
            if current_eff >= 90:
                eff_class, eff_msg = "status-green",  f"{current_eff:.1f}% At or above 90% target ✓"
            elif current_eff >= 85:
                eff_class, eff_msg = "status-yellow", f"{current_eff:.1f}% Getting close to 90% target"
            else:
                eff_class, eff_msg = "status-red",    f"{current_eff:.1f}% Below 80-85% typical range"
 
            st.markdown(
                f'<span class="{eff_class}">Current Efficiency vs Best-Day Ceiling: {eff_msg}</span>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Efficiency % = (median realized CFR) ÷ (your plant's P90 CFR). "
                "A plant at 80-85% efficiency means it's producing 80-85% of what it achieves on its best days. "
                "Getting to 90% is the realistic near-term target."
            )
 
            st.divider()
 
            e1, e2, e3, e4 = st.columns(4)
            e1.metric(
                "Current Efficiency",
                f"{current_eff:.1f}%",
                help="Median CFR ÷ P90 CFR ceiling. The P90 is your own plant's best-day performance, not a theoretical max."
            )
            e2.metric(
                "Gap to 90% Target",
                f"{gap_units:,.0f} gal/day",
                help="How many additional units per day the plant would produce if it reached 90% CFR efficiency. This is the daily opportunity."
            )
            e3.metric(
                "Lost to Downtime",
                f"{wf['downtime_loss']:,.0f} gal/day",
                help="Units lost because downtime takes production lines offline. Calculated as: downtime hours × target CFR rate."
            )
            e4.metric(
                "Lost to Changeovers",
                f"{wf['changeover_loss']:,.0f} gal/day",
                help="Units lost because changeovers (product/SKU switches) take lines offline. Reducing changeover time or frequency directly recovers these units."
            )
 
            st.divider()
 
            # Loss waterfall chart 
            st.markdown("#### Where Capacity Is Going: Loss Waterfall")
            st.caption(
                "Starting from the 90% target capacity, this shows how much of the gap is due to "
                "each type of loss. **CFR Rate Gap** means the line is running, but not as fast "
                "as it could this is where formula complexity, changeover quality, and operator "
                "skill show up."
            )
 
            wf_labels = ["Lost to Downtime", "Lost to Changeovers", "Lost to CFR Rate Gap"]
            wf_values = [wf["downtime_loss"], wf["changeover_loss"], wf["cfr_rate_loss"]]
            wf_total  = sum(wf_values)
 
            if wf_total > 0:
                wf_df = pd.DataFrame({
                    "Loss Type": wf_labels,
                    "Units/Day Lost": [round(v) for v in wf_values],
                    "Share of Total Loss": [f"{v/wf_total*100:.1f}%" for v in wf_values],
                })
                # Horizontal bar chart using st.bar_chart via a transposed df
                wf_chart = pd.DataFrame(
                    {"Units per Day Lost": wf_values},
                    index=wf_labels,
                )
                st.bar_chart(wf_chart)
                st.dataframe(wf_df, use_container_width=True, hide_index=True)
            else:
                st.info("Loss waterfall data unavailable check capacity_daily.csv has hours columns.")
 
            st.divider()
 
            # Per-line summary table ────────────────────────────────────────
            st.markdown("#### Per-Line Capacity Summary: Current vs 90% Target")
            st.caption(
                "**Current** = median realized daily output from history. "
                "**Target 90%** = what output would be if this line ran at 90% of its best-day CFR. "
                "**Best-Day Max** = what the line produces on a P90 day. "
                "The Gap column is the daily unit opportunity per line."
            )
            st.dataframe(gap_data["summary_df"], use_container_width=True, hide_index=True)
 
            st.divider()
 
            # Daily chart: realized vs 90% target vs max ─────────────────
            st.markdown("#### Daily Plant Output: Realized vs Realistic Target vs Best-Day Ceiling")
            st.caption(
                "**Realized Output** = actual daily gallons produced (with day-to-day variability). "
                "**Realized Output** = actual daily plant capacity in gallon-equivalents. "
                "**90% Target** = 90% of the global P90 fill rate × effective hours "
                "the same benchmark used by the efficiency badge above. "
                "**Best-Day Ceiling** = 100% of the global P90 fill rate what the plant achieves on its strongest days. "
                "If the badge reports 74% efficiency, the realized line should sit around 74% of the ceiling."
            )

            daily_df = gap_data["daily_df"].copy()

            if "capacity_at_90pct" in daily_df.columns and "capacity_units" in daily_df.columns:
                n_above   = int((daily_df["capacity_units"] >= daily_df["capacity_at_90pct"]).sum())
                n_total   = len(daily_df)
                pct_above = 100.0 * n_above / n_total if n_total > 0 else 0.0
                if pct_above >= 40:
                    st.success(f"✅ Plant met or exceeded the realistic target on **{pct_above:.0f}%** of days.")
                elif pct_above >= 20:
                    st.warning(f"⚠️ Plant met the realistic target on **{pct_above:.0f}%** of days room to improve.")
                else:
                    st.error(f"🔴 Plant met the realistic target on only **{pct_above:.0f}%** of days structural issue.")

            chart_gap = daily_df.rename(columns={
                "capacity_units":    "Realized Output (gal)",
                "capacity_at_90pct": "Realistic Target (gal)",
                "capacity_at_max":   "Best-Day Ceiling (gal)",
            })[["date", "Realized Output (gal)", "Realistic Target (gal)", "Best-Day Ceiling (gal)"]]
            st.line_chart(chart_gap.set_index("date"))
            st.caption(
                "All values in gallon-equivalents. The 90% target and best-day ceiling are fixed "
                "baselines derived from the global P90 fill rate consistent with the efficiency "
                "badge. The realized line should hover around the efficiency percentage shown above."
            )

            st.divider()

 
            # CFR efficiency trend 
            st.markdown("#### CFR Efficiency Trend Over Time")
            st.caption(
                "Shows how your plant's efficiency (rolling 30-day median CFR as a percentage of "
                "the global P90 best-day ceiling) has moved over time. "
                "Is the plant improving, holding steady, or declining? "
                "The 90% target line shows where you're aiming consistent with the badge above."
            )
 
            if not trend_data.empty:
                trend_chart = trend_data[["date", "efficiency_pct_rolling", "target_90pct_line"]].copy()
                trend_chart = trend_chart.rename(columns={
                    "efficiency_pct_rolling": "Rolling 30-Day Efficiency %",
                    "target_90pct_line":      "90% Target",
                })
                st.line_chart(trend_chart.set_index("date"))
                st.caption(
                    "If the efficiency line is trending up → operational improvements are working. "
                    "Flat → stable but not improving. "
                    "Down → something structural is degrading (more downtime, harder SKU mix, etc.)."
                )
 
            st.divider()
 
            # What does 90% mean in practice? ──────────────────────────────
            st.markdown("#### What Does Getting to 90% Efficiency Actually Mean?")
            st.caption(
                "This translates the gap into concrete, operational language "
                "not just percentages."
            )
 
            # gap_units_90pct is already correctly computed in compute_efficiency_gap():
            # (p90_cfr * 0.90 - med_cfr) * med_eff_hours * n_lines
            # Do NOT recompute from daily_df: the rolling P90 ceiling in daily_df
            # is a per-day smoothed value and its median can fall below realized
            # output, producing a spurious zero gap.
            gap_daily   = float(gap_data["gap_units_90pct"])
            gap_weekly  = gap_daily * 7
            gap_monthly = gap_daily * 30
 
            i1, i2, i3 = st.columns(3)
            i1.metric("Additional gal/day at 90%",   f"{gap_daily:,.0f} gal",   help="Median daily gap to 90% target in gallon-equivalents")
            i2.metric("Additional gal/week at 90%",  f"{gap_weekly:,.0f} gal",  help="7 × daily gap")
            i3.metric("Additional gal/month at 90%", f"{gap_monthly:,.0f} gal", help="30 × daily gap")
 
            st.info(
                f"💡 **For Operations:** To close the gap to 90% efficiency, focus on the biggest "
                f"loss driver above. If **CFR Rate Gap** is largest, the opportunity is in reducing "
                f"changeover time, improving formula consistency, or operator training. "
                f"If **Downtime** is largest, preventive maintenance and rapid response to "
                f"equipment failures will have the highest return."
            )
 
        except Exception as e:
            st.error(f"Efficiency gap analysis failed: {e}")
            st.caption("Check that capacity_daily.csv includes: hours_planned, hours_down, chg_hours, effective_hours, cfr_avg_uh, capacity_units")
