"""
breakdown.py
------------
November vs December bias breakdown (PROTOCOL.md, Section 7). Reporting only:
reads saved predictions, no retraining, never used to select or retune models.

Usage (from repo root):
  python src/breakdown.py

Outputs:
  results/tables/total_nov_dec.csv          floor x version x model x month
  results/tables/total_nov_dec_summary.csv  median over floors per version x model x month
"""

import argparse
import glob
import os
import re

import pandas as pd

from metrics import regression_metrics

MONTHS = {11: "Nov", 12: "Dec"}
KEEP = ["r2", "rmse", "cvrmse_pct", "nmbe_pct"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="results/predictions")
    ap.add_argument("--tables", default="results/tables")
    args = ap.parse_args()

    # CV-selected model per floor x version (rank 1 from the main run), if available
    selected = {}
    all_path = os.path.join(args.tables, "total_all.csv")
    if os.path.exists(all_path):
        allres = pd.read_csv(all_path)
        for _, r in allres[allres.cv_rank == 1].iterrows():
            selected[(int(r.floor), bool(r.lags))] = r.model

    rows = []
    for path in sorted(glob.glob(os.path.join(args.preds, "floor*_*.csv"))):
        m = re.search(r"floor(\d+)_(lags|nolags)\.csv$", os.path.basename(path))
        if not m:
            continue
        floor, lags = int(m.group(1)), m.group(2) == "lags"
        p = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        for month, label in MONTHS.items():
            part = p[p.index.month == month]
            if part.empty:
                continue
            for model in p.columns.drop("actual"):
                s = regression_metrics(part["actual"], part[model])
                rows.append({"floor": floor, "lags": lags, "model": model, "month": label,
                             "hours": len(part), "mean_actual_kwh": part["actual"].mean(),
                             "cv_selected": selected.get((floor, lags)) == model,
                             **{k: s[k] for k in KEEP}})

    if not rows:
        raise SystemExit(f"No prediction files found in {args.preds}")

    out = pd.DataFrame(rows)
    out.round(4).to_csv(os.path.join(args.tables, "total_nov_dec.csv"), index=False)

    summary = (out.groupby(["lags", "model", "month"])[KEEP + ["mean_actual_kwh"]]
               .median().round(3).unstack("month"))
    summary.columns = [f"{k}_{mo}" for k, mo in summary.columns]
    summary.to_csv(os.path.join(args.tables, "total_nov_dec_summary.csv"))

    pd.set_option("display.width", 200)
    cols = [f"{k}_{mo}" for k in ("nmbe_pct", "cvrmse_pct", "r2") for mo in ("Nov", "Dec")]
    print("=== Median over floors: November vs December (reporting only) ===")
    print(summary[cols].to_string())
    load = out.drop_duplicates(["floor", "month"]).pivot(index="floor", columns="month",
                                                         values="mean_actual_kwh").round(1)
    print("\nMean actual load (kWh/h) per floor:\n" + load.to_string())
    if selected:
        sel = out[out.cv_selected].groupby(["lags", "month"])[["nmbe_pct", "r2"]].median().round(3)
        print("\nCV-selected model per floor, median over floors:\n" + sel.to_string())
    print(f"\nSaved -> {args.tables}/total_nov_dec*.csv")


if __name__ == "__main__":
    main()
