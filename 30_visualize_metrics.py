#!/usr/bin/env python3
"""
CREATED BY LISETTE KAMPER_HINSON FOR MS THESIS @ WAKE FOREST 
30_visualize_metrics.py
 
Goal
  Create basic diagnostic plots from out/capacity_daily.csv.
 
Outputs
  - out/plots/cfr_by_line.png
  - out/plots/capacity_by_line.png
  - out/plots/plant_capacity.png
"""
 
from __future__ import annotations
 
import argparse
from pathlib import Path
 
import pandas as pd
import matplotlib.pyplot as plt
 
 
def plot_by_line(df: pd.DataFrame, ycol: str, title: str, ylabel: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for ln, g in df.groupby("line"):
        g = g.sort_values("date")
        ax.plot(g["date"], g[ycol], label=str(ln))
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
 
 
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap_csv", default="out/capacity_daily.csv")
    ap.add_argument("--out_dir", default="out/plots")
    args = ap.parse_args()
 
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
 
    df = pd.read_csv(args.cap_csv)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
 
    if "cfr_avg_uh" in df.columns:
        plot_by_line(
            df, "cfr_avg_uh",
            "Daily CFR (good units/hour) by Line",
            "CFR (good units/hour)",
            f"{args.out_dir}/cfr_by_line.png"
        )
 
    plot_by_line(
        df, "capacity_units",
        "Daily Capacity (good units/day) by Line",
        "Capacity (good units/day)",
        f"{args.out_dir}/capacity_by_line.png"
    )
 
    plant = df.groupby("date", as_index=False)["capacity_units"].sum().sort_values("date")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(plant["date"], plant["capacity_units"])
    ax.set_title("Plant Total Capacity (good units/day)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Capacity (good units/day)")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/plant_capacity.png", dpi=200)
    plt.close(fig)
 
    print(f"Saved plots to {args.out_dir}")
 
 
if __name__ == "__main__":
    main()

