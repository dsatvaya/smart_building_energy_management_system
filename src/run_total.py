"""
run_total.py
------------
Experiment 1: predict hourly FLOOR TOTAL energy.
Loops floors x {with lags, without lags}, runs all 5 models, saves tables.

Floors used: 2, 3, 4, 5, 7
  (Floor 1 excluded: no indoor sensors. Floor 6 excluded: source file ends 22 Oct 2019.)

Split: one SHARED calendar test period for every floor (default 1 Nov - 31 Dec 2019).
Training = all usable hours before it; tuning uses training-only TimeSeriesSplit CV.
Models are ranked by training CV only: per floor by CV RMSE (cv_rank), across floors by
normalized CV(RMSE) % (CV RMSE / mean training load) and mean rank. The test period is only
used to report final performance.

Usage (from repo root):
  python src/run_total.py
  python src/run_total.py --floors 7          # quick single-floor run

Outputs:
  results/tables/total_all.csv             every floor x model x version, all metrics
  results/tables/total_summary.csv         paper-style table: median across floors
  results/tables/total_lag_vs_nolag.csv    test R2 with vs without lags
  results/tables/total_skipped.csv         floors skipped (too few train/test hours or missing)
  results/predictions/floorN_lags|nolags.csv   hourly actual vs predicted (for plots)
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from features import date_split, get_frame, load_clean, make_features
from models import run_experiment

FLOORS = [2, 3, 4, 5, 7]
TEST_START = "2019-11-01"
MIN_TRAIN, MIN_TEST = 1600, 400   # usable hours needed before / after TEST_START
KEY = ["r2", "mse", "rmse", "vaf_pct", "a20_pct", "ioa", "cvrmse_pct", "nmbe_pct"]


def usable_split(clean_path, test_start):
    """Usable (train, test) hours: valid floor_total AND full lag history.
    Same rows for both lag versions."""
    X, y = make_features(get_frame(load_clean(clean_path), "floor_total"), lags=True)
    X_tr, X_te, _, _ = date_split(X, y, test_start)
    return len(X_tr), len(X_te)


def save_predictions(path, clean_path, use_lags, fitted, test_start):
    """Hourly actual vs each model's prediction on the shared test period."""
    X, y = make_features(get_frame(load_clean(clean_path), "floor_total"), lags=use_lags)
    _, X_te, _, y_te = date_split(X, y, test_start)
    out = pd.DataFrame({"actual": y_te})
    for name, search in fitted.items():
        out[name] = np.clip(search.best_estimator_.predict(X_te), 0.0, None)
    out.to_csv(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", default="data/clean")
    ap.add_argument("--floors", nargs="+", type=int, default=FLOORS)
    ap.add_argument("--test-start", default=TEST_START)
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--preds", default="results/predictions")
    args = ap.parse_args()
    os.makedirs(args.tables, exist_ok=True)
    os.makedirs(args.preds, exist_ok=True)

    rows, skipped = [], []
    for floor in args.floors:
        clean_path = os.path.join(args.clean_dir, f"floor{floor}_hourly.csv")
        if not os.path.exists(clean_path):
            reason = "clean file not found"
        else:
            n_tr, n_te = usable_split(clean_path, args.test_start)
            reason = (f"too few usable hours (train {n_tr} < {MIN_TRAIN} or test {n_te} < {MIN_TEST})"
                      if n_tr < MIN_TRAIN or n_te < MIN_TEST else None)
            if reason is None:
                print(f"Floor {floor}: usable train {n_tr} h, test {n_te} h [OK]")
        if reason:
            print(f"Floor {floor}: SKIPPED - {reason}")
            skipped.append({"floor": floor, "reason": reason})
            continue

        for use_lags in (True, False):
            tag = "lags" if use_lags else "nolags"
            t = time.time()
            res, fitted = run_experiment(clean_path, "floor_total", use_lags,
                                         test_start=args.test_start)
            res.insert(0, "floor", floor)
            rows.append(res)
            save_predictions(os.path.join(args.preds, f"floor{floor}_{tag}.csv"),
                             clean_path, use_lags, fitted, args.test_start)
            best = res.loc[res["cv_rmse_mean"].idxmin()]   # selection: training CV only
            print(f"  {tag:6s} | best by CV RMSE: {best.model} (CV {best.cv_rmse_mean:.2f}) "
                  f"-> test R2 {best.test_r2:.3f} | {time.time() - t:.0f}s")
        # checkpoint: a crash halfway keeps finished floors
        pd.concat(rows).to_csv(os.path.join(args.tables, "total_all.csv"), index=False)

    if skipped:
        pd.DataFrame(skipped).to_csv(os.path.join(args.tables, "total_skipped.csv"), index=False)
    if not rows:
        raise SystemExit("No floor had enough usable hours - nothing to report.")

    allres = pd.concat(rows, ignore_index=True)
    # per-floor rank by training CV RMSE (1 = best) - scale-free across floors
    allres["cv_rank"] = allres.groupby(["floor", "lags"])["cv_rmse_mean"].rank(method="min")
    # ASHRAE Guideline 14 hourly CALIBRATION thresholds, used here only as a reference point
    # for forecast accuracy - meeting them is not an ASHRAE "pass" for a forecasting model.
    allres["within_ashrae_calib_thresholds"] = (
        (allres.test_cvrmse_pct <= 30) & (allres.test_nmbe_pct.abs() <= 10))
    allres.to_csv(os.path.join(args.tables, "total_all.csv"), index=False)

    # paper-style table: per model x version, training vs testing (median over floors)
    agg = {f"{s}_{k}": "median" for s in ("train", "test") for k in KEY}
    agg["cv_cvrmse_pct"] = "median"
    agg["cv_rank"] = "mean"
    agg["within_ashrae_calib_thresholds"] = "mean"
    summary = allres.groupby(["lags", "model"]).agg(agg).round(3)
    summary["within_ashrae_calib_thresholds"] = (summary["within_ashrae_calib_thresholds"] * 100).round(0)
    summary = (summary.rename(columns={"within_ashrae_calib_thresholds": "pct_floors_within_ashrae_calib"})
               .rename(columns={"cv_cvrmse_pct": "median_cv_cvrmse_pct", "cv_rank": "mean_cv_rank"})
               .sort_values(["lags", "median_cv_cvrmse_pct"]))
    summary.to_csv(os.path.join(args.tables, "total_summary.csv"))

    lag_cmp = allres.pivot_table(index=["floor", "model"], columns="lags", values="test_r2").round(3)
    lag_cmp = lag_cmp.rename(columns={False: "r2_nolags", True: "r2_lags"})
    lag_cmp.to_csv(os.path.join(args.tables, "total_lag_vs_nolag.csv"))

    pd.set_option("display.width", 200)
    print(f"\n=== Across floors (ranked by median training CV(RMSE) %; test = {args.test_start} onward) ===")
    print(summary[["median_cv_cvrmse_pct", "mean_cv_rank"] + [f"test_{k}" for k in ("r2", "rmse", "cvrmse_pct", "nmbe_pct", "ioa")]
                  + ["pct_floors_within_ashrae_calib"]].to_string())
    print("\nReference: ASHRAE Guideline 14 hourly calibration thresholds, CV(RMSE) <= 30% and |NMBE| <= 10%.")
    print("These are calibration criteria, shown for comparison only - not an ASHRAE pass for forecasts.")
    print(f"\nSaved -> {args.tables}/ and {args.preds}/")


if __name__ == "__main__":
    main()
