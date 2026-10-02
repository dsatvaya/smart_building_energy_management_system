"""
run_rolling.py
--------------
Supplementary rolling-origin check (PROTOCOL.md, Section 7).

For each origin (1 Aug, 1 Sep, 1 Oct, 1 Nov, 1 Dec 2019):
  train on all usable hours before the origin, tune with the same CV + grids,
  evaluate on the following calendar month.
With-lags feature set only (bounds compute). Selection still uses training CV only.

Purpose: show whether over-prediction (positive NMBE) is specific to Nov-Dec
or present every month.

Usage (from repo root):
  python src/run_rolling.py

Outputs:
  results/tables/rolling_all.csv        floor x origin x model, all metrics
  results/tables/rolling_by_month.csv   median over floors per month x model
  results/tables/rolling_skipped.csv    floor/origin pairs with too few hours
"""

import argparse
import os
import time

import pandas as pd

from features import date_split, get_frame, load_clean, make_features
from models import run_experiment
from run_total import FLOORS, MIN_TEST, MIN_TRAIN

ORIGINS = ["2019-08-01", "2019-09-01", "2019-10-01", "2019-11-01", "2019-12-01"]


def month_end(origin):
    return str((pd.Timestamp(origin) + pd.offsets.MonthBegin(1)).date())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", default="data/clean")
    ap.add_argument("--floors", nargs="+", type=int, default=FLOORS)
    ap.add_argument("--tables", default="results/tables")
    args = ap.parse_args()
    os.makedirs(args.tables, exist_ok=True)

    rows, skipped = [], []
    for floor in args.floors:
        clean_path = os.path.join(args.clean_dir, f"floor{floor}_hourly.csv")
        if not os.path.exists(clean_path):
            skipped.append({"floor": floor, "origin": None, "reason": "clean file not found"})
            print(f"Floor {floor}: SKIPPED - clean file not found")
            continue
        X, y = make_features(get_frame(load_clean(clean_path), "floor_total"), lags=True)

        for origin in ORIGINS:
            end = month_end(origin)
            X_tr, X_te, _, _ = date_split(X, y, origin, end)
            if len(X_tr) < MIN_TRAIN or len(X_te) < MIN_TEST:
                reason = f"train {len(X_tr)} / eval {len(X_te)} hours below {MIN_TRAIN}/{MIN_TEST}"
                skipped.append({"floor": floor, "origin": origin, "reason": reason})
                print(f"Floor {floor} {origin}: SKIPPED - {reason}")
                continue
            t = time.time()
            res, _ = run_experiment(clean_path, "floor_total", True,
                                    test_start=origin, test_end=end)
            res.insert(0, "origin", origin)
            res.insert(0, "floor", floor)
            rows.append(res)
            best = res.loc[res["cv_rmse_mean"].idxmin()]   # selection: training CV only
            print(f"Floor {floor} {origin} | train {len(X_tr)} eval {len(X_te)} | "
                  f"best by CV: {best.model} | {time.time() - t:.0f}s")
            pd.concat(rows).to_csv(os.path.join(args.tables, "rolling_all.csv"), index=False)

    if skipped:
        pd.DataFrame(skipped).to_csv(os.path.join(args.tables, "rolling_skipped.csv"), index=False)
    if not rows:
        raise SystemExit("No floor/origin had enough usable hours.")

    allres = pd.concat(rows, ignore_index=True)
    allres["cv_rank"] = allres.groupby(["floor", "origin"])["cv_rmse_mean"].rank(method="min")
    allres.to_csv(os.path.join(args.tables, "rolling_all.csv"), index=False)

    by_month = (allres.groupby(["origin", "model"])
                .agg(median_cv_cvrmse_pct=("cv_cvrmse_pct", "median"),
                     mean_cv_rank=("cv_rank", "mean"),
                     test_r2=("test_r2", "median"),
                     test_cvrmse_pct=("test_cvrmse_pct", "median"),
                     test_nmbe_pct=("test_nmbe_pct", "median"),
                     floors=("floor", "nunique"))
                .round(3).sort_values(["origin", "median_cv_cvrmse_pct"]))
    by_month.to_csv(os.path.join(args.tables, "rolling_by_month.csv"))

    pd.set_option("display.width", 200)
    print("\n=== Rolling origin: median over floors, per evaluation month (with lags) ===")
    print(by_month.to_string())
    print(f"\nSaved -> {args.tables}/rolling_*.csv")


if __name__ == "__main__":
    main()
