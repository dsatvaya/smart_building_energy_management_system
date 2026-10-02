"""
run_experiments.py
------------------
Sweeps every zone x component in one or more CU-BEMS floor files, trains all
baseline models on each, and collates a single comparison table.

This produces the headline result for the mid-term: a model-vs-model table
across the whole building, directly comparable to Table 6 of Sari et al. (2023).

Usage:
  python run_experiments.py --csv data/2019Floor6.csv
  python run_experiments.py --csv data/2019Floor2.csv data/2019Floor6.csv --components ac light
  python run_experiments.py --synthetic            # smoke test
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd

from bems_pipeline import (
    load_floor, build_frame, make_features, temporal_split,
    build_models, evaluate, available_zones, component_columns,
    synthetic_floor_csv, predict_nonneg,
)

warnings.filterwarnings("ignore")

PAPER_MAPE = 22.01  # Sari et al. (2023), average across floors


def run_one(df, zone, component, test_frac, min_samples=500):
    """Train all models on one zone/component. Returns list of metric rows."""
    frame = build_frame(df, zone, component)
    X, y = make_features(frame)

    if len(X) < min_samples:
        raise ValueError(f"only {len(X)} usable hours after cleaning")
    if y.sum() <= 0:
        raise ValueError("target is all zeros — meter likely dead")

    X_tr, X_te, y_tr, y_te = temporal_split(X, y, test_frac)

    rows = []
    for name, model in build_models(len(X_tr)).items():
        model.fit(X_tr, y_tr)
        m = evaluate(y_te, predict_nonneg(model, X_te))
        m.update({"model": name, "n_train": len(X_tr), "n_test": len(X_te)})
        rows.append(m)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", help="one or more floor CSVs")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--components", nargs="+", default=["ac", "light", "plug"],
                    choices=["ac", "light", "plug"])
    ap.add_argument("--zones", nargs="+", type=int, default=None,
                    help="default: every zone found in the file")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.synthetic:
        paths = [synthetic_floor_csv(os.path.join(args.outdir, "_synthetic_floor.csv"))]
        print("[!] SYNTHETIC MODE — smoke test only, not reportable results.\n")
    elif args.csv:
        paths = args.csv
    else:
        ap.error("pass --csv <files> or --synthetic")

    all_rows, skipped = [], []

    for path in paths:
        floor_label = os.path.splitext(os.path.basename(path))[0]
        print(f"\n{'=' * 70}\n{floor_label}\n{'=' * 70}")
        df = load_floor(path)

        zones = args.zones or available_zones(df.columns)
        for zone in zones:
            for comp in args.components:
                if not component_columns(df.columns, zone, comp):
                    continue  # this zone has no meter of that type
                try:
                    rows = run_one(df, zone, comp, args.test_frac)
                except Exception as e:
                    skipped.append((floor_label, zone, comp, str(e)))
                    print(f"  skip  z{zone}/{comp:5s} — {e}")
                    continue

                for r in rows:
                    r.update({"floor": floor_label, "zone": zone, "component": comp})
                all_rows.extend(rows)

                best = min(rows, key=lambda r: r["CV_RMSE_%"])
                flag = "PASS" if best["CV_RMSE_%"] <= 30 else "fail"
                print(f"  ok    z{zone}/{comp:5s} — best: {best['model']:22s} "
                      f"CV(RMSE) {best['CV_RMSE_%']:6.2f}% [{flag}]  R2 {best['R2']:.3f}  "
                      f"MAPE {best['MAPE_%']:7.2f}%  lowload {best['low_load_hours_%']:4.1f}%")

        # checkpoint after every floor — a 3-hour run should survive a crash
        if all_rows:
            pd.DataFrame(all_rows).to_csv(
                os.path.join(args.outdir, "all_results_partial.csv"), index=False)
            print(f"  [checkpoint] {len(all_rows)} rows saved")

    if not all_rows:
        print("\nNo results produced. Check the file path and column names.")
        return

    # ── master table ──────────────────────────────────────────
    cols = ["floor", "zone", "component", "model", "CV_RMSE_%", "NMBE_%", "R2",
            "MAPE_%", "sMAPE_%", "RMSE_kWh", "TotalEnergyRelErr_%",
            "mean_kWh", "low_load_hours_%", "n_train", "n_test"]
    results = pd.DataFrame(all_rows)[cols]
    master = os.path.join(args.outdir, "all_results.csv")
    results.to_csv(master, index=False)

    # ── model-level summary: the slide-worthy table ────────────
    results["ashrae_pass"] = results["CV_RMSE_%"] <= 30
    summary = (results
               .groupby("model")
               .agg(median_CV_RMSE=("CV_RMSE_%", "median"),
                    median_NMBE=("NMBE_%", "median"),
                    median_R2=("R2", "median"),
                    ashrae_pass_rate=("ashrae_pass", "mean"),
                    median_MAPE=("MAPE_%", "median"),
                    n_cases=("CV_RMSE_%", "size"))
               .round(3)
               .sort_values("median_CV_RMSE"))
    summary["ashrae_pass_rate"] = (summary["ashrae_pass_rate"] * 100).round(1)

    n_cases = results.groupby(["floor", "zone", "component"]).ngroups
    print(f"\n{'=' * 78}\nSUMMARY ACROSS {n_cases} ZONE/COMPONENT CASES\n{'=' * 78}")
    print("Primary metric: CV(RMSE), ASHRAE Guideline 14. Hourly threshold <= 30%.\n")
    print(summary.to_string())
    summary.to_csv(os.path.join(args.outdir, "model_summary.csv"))

    best_model = summary.index[0]
    best_cv = summary.iloc[0]["median_CV_RMSE"]
    pass_rate = summary.iloc[0]["ashrae_pass_rate"]
    print(f"\nBest model: {best_model}")
    print(f"  median CV(RMSE) {best_cv}%  |  meets ASHRAE threshold in {pass_rate}% of cases")
    print(f"  median MAPE {summary.iloc[0]['median_MAPE']}%   "
          f"(Sari et al. 2023 k-NN baseline: {PAPER_MAPE}%)")

    # ── why MAPE misleads here ─────────────────────────────────
    worst = results.nlargest(5, "MAPE_%")[
        ["zone", "component", "model", "MAPE_%", "CV_RMSE_%", "R2", "low_load_hours_%"]]
    print(f"\nCases where MAPE misleads (high MAPE, high R2, many near-zero hours):")
    print(worst.to_string(index=False))
    worst.to_csv(os.path.join(args.outdir, "mape_failure_cases.csv"), index=False)

    # ── per-component breakdown ────────────────────────────────
    by_comp = (results.groupby(["component", "model"])["CV_RMSE_%"]
               .median().unstack().round(2))
    print(f"\nMedian CV(RMSE) by component:\n{by_comp.to_string()}")
    by_comp.to_csv(os.path.join(args.outdir, "cvrmse_by_component.csv"))

    # ── per-floor breakdown: directly comparable to Table 6 of Sari et al. ──
    if results["floor"].nunique() > 1:
        by_floor = (results.groupby(["floor", "model"])
                    .agg(CV_RMSE=("CV_RMSE_%", "median"),
                         R2=("R2", "median"),
                         MAPE=("MAPE_%", "median"),
                         cases=("R2", "size"))
                    .round(2))
        print(f"\nPer-floor results (compare with Sari et al. Table 6):\n{by_floor.to_string()}")
        by_floor.to_csv(os.path.join(args.outdir, "results_by_floor.csv"))

        # where does the AC problem show up hardest?
        ac = results[results["component"] == "ac"]
        if len(ac):
            ac_summary = (ac.groupby("floor")
                          .agg(median_CV_RMSE=("CV_RMSE_%", "median"),
                               median_lowload=("low_load_hours_%", "median"),
                               cases=("R2", "size"))
                          .round(2))
            print(f"\nAC loads by floor (low-load % drives the difficulty):\n{ac_summary.to_string()}")
            ac_summary.to_csv(os.path.join(args.outdir, "ac_difficulty_by_floor.csv"))

    if skipped:
        pd.DataFrame(skipped, columns=["floor", "zone", "component", "reason"]).to_csv(
            os.path.join(args.outdir, "skipped.csv"), index=False)
        print(f"\n{len(skipped)} case(s) skipped — see skipped.csv")

    print(f"\nWritten: {master}, model_summary.csv, mape_by_component.csv")


if __name__ == "__main__":
    main()
