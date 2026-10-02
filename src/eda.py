"""
eda.py
------
Descriptive analysis (report Sections 4.1-4.3), on exactly the hours used for modelling.

  Table 4  descriptive statistics      -> eda_stats_combined.csv   (all floors pooled)
  Table 5  association with floor_total -> eda_corr_combined.csv    (median of per-floor values)
           Pearson r (straight-line) + mutual information (any shape, e.g. on/off by hour)
  Composition of floor_total (AC / light / plug shares) -> eda_composition.csv
  Fig. 2   inputs vs floor_total        -> fig2_scatter.png
  Appendix per-floor versions           -> eda_stats_by_floor.csv, eda_corr_by_floor.csv

naive_sem = ordinary standard error of the mean (std / sqrt(n)). It treats hours as
independent, but consecutive hours are strongly related, so it understates the true
uncertainty. Descriptive only - do not use it for confidence intervals or significance tests.

Pearson r and mutual information are descriptive, in-sample estimates of association.
MI indicates a nonlinear association; it is not evidence of out-of-sample predictive
strength (that is assessed in the sensitivity analysis).

Combined values = median of each floor's own value, so differences in floor size are not
mistaken for relationships between variables. Components (AC, light, plug) are PARTS of
floor_total, so they are shown in the composition table, not the association table.

Usage (from repo root):
  python src/eda.py
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from features import get_frame, load_clean, make_features
from run_total import FLOORS

COMPONENTS = ["floor_ac", "floor_light", "floor_plug"]
CORR_VARS = ["temp_c", "rh_pct", "lux", "hour", "dow", "is_weekend", "is_holiday",
             "lag_1", "lag_24", "lag_168", "roll_24"]
DISCRETE = {"hour", "dow", "is_weekend", "is_holiday"}
STAT_VARS = ["temp_c", "rh_pct", "lux"] + COMPONENTS + ["floor_total"]
SCATTER = [("temp_c", "Indoor temperature (\u00b0C)"), ("rh_pct", "Relative humidity (%)"),
           ("lux", "Light level (lux)"), ("hour", "Hour of day"),
           ("lag_1", "Energy 1 h earlier (kWh)"), ("lag_24", "Energy 24 h earlier (kWh)")]


def floor_data(clean_path):
    """Modelling rows (same as run_total.py) + floor components for description."""
    clean = load_clean(clean_path)
    X, y = make_features(get_frame(clean, "floor_total"), lags=True)
    d = X.copy()
    d["floor_total"] = y
    for c in COMPONENTS:
        d[c] = clean.loc[X.index, c]
    return d


def describe(d):
    s = d[STAT_VARS]
    return pd.DataFrame({
        "n": s.count(), "mean": s.mean(), "naive_sem": s.sem(), "std_dev": s.std(),
        "variance": s.var(), "kurtosis": s.kurt(), "skewness": s.skew(),
        "min": s.min(), "max": s.max(),
    }).round(3)


def mutual_info(d):
    mi = mutual_info_regression(d[CORR_VARS], d["floor_total"], random_state=42,
                                discrete_features=[c in DISCRETE for c in CORR_VARS])
    return pd.Series(mi, index=CORR_VARS)


def composition(d):
    tot = d["floor_total"].sum()
    return pd.Series({c.replace("floor_", "") + "_share_pct": 100 * d[c].sum() / tot for c in COMPONENTS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", default="data/clean")
    ap.add_argument("--floors", nargs="+", type=int, default=FLOORS)
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--figures", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.tables, exist_ok=True)
    os.makedirs(args.figures, exist_ok=True)

    data = {f: floor_data(os.path.join(args.clean_dir, f"floor{f}_hourly.csv")) for f in args.floors}
    pooled = pd.concat(data.values(), keys=data.keys(), names=["floor", "Date"]).reset_index()

    # Table 4: pooled descriptive statistics; appendix: per floor
    describe(pooled).to_csv(os.path.join(args.tables, "eda_stats_combined.csv"))
    by_floor = pd.concat({f: describe(d) for f, d in data.items()}, names=["floor", "variable"])
    by_floor.to_csv(os.path.join(args.tables, "eda_stats_by_floor.csv"))

    # Table 5: correlation with floor_total, per floor and median across floors
    corr = pd.DataFrame({f: d[CORR_VARS].corrwith(d["floor_total"]) for f, d in data.items()})
    corr.columns = [f"floor{f}" for f in corr.columns]
    mi = pd.DataFrame({f"floor{f}": mutual_info(d) for f, d in data.items()})
    pd.concat({"pearson_r": corr, "mutual_info": mi}, axis=1).round(3).to_csv(
        os.path.join(args.tables, "eda_corr_by_floor.csv"))
    combined = pd.DataFrame({"median_r": corr.median(axis=1), "min_r": corr.min(axis=1),
                             "max_r": corr.max(axis=1), "median_mi": mi.median(axis=1)}).round(3)
    combined = combined.sort_values("median_mi", ascending=False)
    combined.to_csv(os.path.join(args.tables, "eda_corr_combined.csv"))

    comp = pd.DataFrame({f"floor{f}": composition(d) for f, d in data.items()})
    comp["all_floors"] = composition(pooled)
    comp.round(1).to_csv(os.path.join(args.tables, "eda_composition.csv"))

    # Figure 2: inputs vs floor_total, coloured by floor
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (col, label) in zip(axes.ravel(), SCATTER):
        for f, d in data.items():
            ax.scatter(d[col], d["floor_total"], s=3, alpha=0.25, label=f"Floor {f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Floor total energy (kWh)")
        ax.set_title(f"{label} vs energy (median r = {combined.loc[col, 'median_r']:.2f})", fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(data), markerscale=4)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(args.figures, "fig2_scatter.png"), dpi=200)

    pd.set_option("display.width", 200)
    print(f"Rows used: {len(pooled)} floor-hours across floors {args.floors}\n")
    print("=== Table 4: descriptive statistics (all floors pooled) ===")
    print(describe(pooled).to_string())
    print("\n=== Table 5: association with floor_total (median over floors, sorted by MI) ===")
    print(combined.to_string())
    print("\n=== Composition of floor_total (% of energy) ===")
    print(comp.round(1).to_string())
    print(f"\nSaved -> {args.tables}/eda_*.csv and {args.figures}/fig2_scatter.png")


if __name__ == "__main__":
    main()
