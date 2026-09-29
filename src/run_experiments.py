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

                best = min(rows, key=lambda r: r["RMSE_kWh"])
                print(f"  ok    z{zone}/{comp:5s} — best: {best['model']:22s} "
                      f"MAPE {best['MAPE_%']:6.2f}%  R2 {best['R2']:.3f}")

    if not all_rows:
        print("\nNo results produced. Check the file path and column names.")
        return

    # ── master table ──────────────────────────────────────────
    cols = ["floor", "zone", "component", "model", "MAPE_%", "sMAPE_%",
            "RMSE_kWh", "R2", "TotalEnergyRelErr_%", "n_train", "n_test"]
    results = pd.DataFrame(all_rows)[cols]
    master = os.path.join(args.outdir, "all_results.csv")
    results.to_csv(master, index=False)

    # ── model-level summary: the slide-worthy table ────────────
    summary = (results
               .groupby("model")
               .agg(median_MAPE=("MAPE_%", "median"),
                    mean_MAPE=("MAPE_%", "mean"),
                    median_sMAPE=("sMAPE_%", "median"),
                    median_R2=("R2", "median"),
                    n_cases=("MAPE_%", "size"))
               .round(2)
               .sort_values("median_MAPE"))

    print(f"\n{'=' * 70}\nSUMMARY ACROSS {results.groupby(['floor','zone','component']).ngroups} ZONE/COMPONENT CASES\n{'=' * 70}")
    print(summary.to_string())
    summary.to_csv(os.path.join(args.outdir, "model_summary.csv"))

    best_model = summary.index[0]
    best_mape = summary.iloc[0]["median_MAPE"]
    print(f"\nBest model: {best_model}  (median MAPE {best_mape}%)")
    print(f"Sari et al. (2023) k-NN baseline: MAPE {PAPER_MAPE}%")
    if best_mape < PAPER_MAPE:
        delta = (PAPER_MAPE - best_mape) / PAPER_MAPE * 100
        print(f"→ {delta:.1f}% relative improvement over the published baseline.")
    else:
        print("→ Not yet beating the published baseline. Check feature set and tuning.")

    # ── per-component breakdown ────────────────────────────────
    by_comp = (results.groupby(["component", "model"])["MAPE_%"]
               .median().unstack().round(2))
    print(f"\nMedian MAPE by component:\n{by_comp.to_string()}")
    by_comp.to_csv(os.path.join(args.outdir, "mape_by_component.csv"))

    if skipped:
        pd.DataFrame(skipped, columns=["floor", "zone", "component", "reason"]).to_csv(
            os.path.join(args.outdir, "skipped.csv"), index=False)
        print(f"\n{len(skipped)} case(s) skipped — see skipped.csv")

    print(f"\nWritten: {master}, model_summary.csv, mape_by_component.csv")


if __name__ == "__main__":
    main()
