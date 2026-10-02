"""
explain.py
----------
Sensitivity analysis (report Section 4.6 / Table 7) for the Random Forest, the overall
CV-selected model, on every floor, with and without lags. Reporting only.

No retuning: each RF is rebuilt from the FROZEN run's chosen hyperparameters
(best_params in total_all.csv) and refit on the same training hours. RF is deterministic
(fixed seed), so the rebuilt model must reproduce the frozen run's saved evaluation
predictions; the script stops if it does not.

Measures
  rf_importance_pct   impurity-based importance (training; biased towards continuous,
                      high-cardinality features - read together with the others)
  shap_pct            mean |SHAP value| share, evaluation period (fixed-seed sample)
  perm_r2_drop        drop in evaluation R2 when one feature is shuffled (out-of-sample)
  group_perm_r2_drop  drop in evaluation R2 when a GROUP of related features is shuffled
                      together (e.g. hour + hour_sin + hour_cos). Related features carry
                      the same information, so shuffling one alone understates the group.

Usage (from repo root):
  python src/explain.py

Outputs:
  results/tables/explain_by_floor.csv         floor x version x feature
  results/tables/explain_summary.csv          median over floors (Table 7)
  results/tables/explain_groups_by_floor.csv  grouped permutation per floor
  results/tables/explain_groups_summary.csv   grouped permutation, median over floors
"""

import argparse
import ast
import os

import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

from features import date_split, get_frame, load_clean, make_features
from models import model_searches
from run_total import FLOORS, TEST_START

SEED = 42
N_REPEATS = 10
GROUPS = {
    "hour (hour, hour_sin, hour_cos)": ["hour", "hour_sin", "hour_cos"],
    "day of week (dow, dow_sin, dow_cos, is_weekend)": ["dow", "dow_sin", "dow_cos", "is_weekend"],
    "holiday": ["is_holiday"],
    "temperature": ["temp_c"],
    "humidity": ["rh_pct"],
    "lux": ["lux"],
    "past energy (lag_1, lag_24, lag_168, roll_24)": ["lag_1", "lag_24", "lag_168", "roll_24"],
}


def rebuild_rf(X_tr, y_tr, params):
    """Frozen RF configuration with the frozen run's chosen hyperparameters."""
    est = clone(model_searches(len(X_tr))["RandomForest"].estimator)
    est.set_params(**params)
    return est.fit(X_tr, y_tr)


def grouped_permutation(model, X, y, rng):
    base = r2_score(y, np.clip(model.predict(X), 0, None))
    out = {}
    for name, cols in GROUPS.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        drops = []
        for _ in range(N_REPEATS):
            Xp = X.copy()
            order = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[order]          # shuffle the group jointly
            drops.append(base - r2_score(y, np.clip(model.predict(Xp), 0, None)))
        out[name] = (np.mean(drops), np.std(drops))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", default="data/clean")
    ap.add_argument("--floors", nargs="+", type=int, default=FLOORS)
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--preds", default="results/predictions")
    ap.add_argument("--shap-sample", type=int, default=1000)
    args = ap.parse_args()

    allres = pd.read_csv(os.path.join(args.tables, "total_all.csv"))   # frozen main run
    rows, grows = [], []
    for floor in args.floors:
        clean_path = os.path.join(args.clean_dir, f"floor{floor}_hourly.csv")
        frame = get_frame(load_clean(clean_path), "floor_total")
        for lags in (True, False):
            tag = "lags" if lags else "nolags"
            X, y = make_features(frame, lags=lags)
            X_tr, X_te, y_tr, y_te = date_split(X, y, TEST_START)
            meta = allres.query("floor == @floor and lags == @lags and model == 'RandomForest'").iloc[0]
            model = rebuild_rf(X_tr, y_tr, ast.literal_eval(meta.best_params))

            # reproduction check against the frozen run's saved predictions
            pred = np.clip(model.predict(X_te), 0, None)
            frozen = pd.read_csv(os.path.join(args.preds, f"floor{floor}_{tag}.csv"),
                                 parse_dates=["Date"], index_col="Date")["RandomForest"]
            if not (frozen.index.equals(X_te.index) and np.allclose(frozen.to_numpy(), pred)):
                raise SystemExit(f"Floor {floor} {tag}: rebuilt RF does not reproduce the frozen "
                                 f"predictions - stopping.")

            rf = model.named_steps["model"]
            imp = pd.Series(rf.feature_importances_ * 100, index=X.columns)

            Xs = X_te.sample(min(args.shap_sample, len(X_te)), random_state=SEED)
            sv = np.abs(shap.TreeExplainer(rf).shap_values(Xs.to_numpy())).mean(axis=0)
            shap_pct = pd.Series(100 * sv / sv.sum(), index=X.columns)

            perm = permutation_importance(model, X_te, y_te, scoring="r2",
                                          n_repeats=N_REPEATS, random_state=SEED)
            for i, feat in enumerate(X.columns):
                rows.append({"floor": floor, "lags": lags, "feature": feat,
                             "rf_importance_pct": imp[feat], "shap_pct": shap_pct[feat],
                             "perm_r2_drop": perm.importances_mean[i],
                             "perm_r2_drop_std": perm.importances_std[i]})

            for name, (m, s) in grouped_permutation(model, X_te, y_te,
                                                    np.random.default_rng(SEED)).items():
                grows.append({"floor": floor, "lags": lags, "group": name,
                              "group_perm_r2_drop": m, "group_perm_r2_drop_std": s})
            print(f"Floor {floor} {tag:6s}: rebuilt RF reproduces frozen predictions [OK]; "
                  f"eval R2 {r2_score(y_te, pred):.3f}")

    by_floor = pd.DataFrame(rows)
    by_floor.round(4).to_csv(os.path.join(args.tables, "explain_by_floor.csv"), index=False)
    summary = (by_floor.groupby(["lags", "feature"])
               [["rf_importance_pct", "shap_pct", "perm_r2_drop"]].median().round(3)
               .sort_values(["lags", "shap_pct"], ascending=[False, False]))
    summary.to_csv(os.path.join(args.tables, "explain_summary.csv"))

    groups = pd.DataFrame(grows)
    groups.round(4).to_csv(os.path.join(args.tables, "explain_groups_by_floor.csv"), index=False)
    gsum = (groups.groupby(["lags", "group"])["group_perm_r2_drop"]
            .agg(median="median", min="min", max="max").round(3)
            .sort_values(["lags", "median"], ascending=[False, False]))
    gsum.to_csv(os.path.join(args.tables, "explain_groups_summary.csv"))

    pd.set_option("display.width", 200)
    print("\n=== Table 7: Random Forest feature importance (median over floors) ===")
    print(summary.to_string())
    print("\n=== Grouped permutation: drop in evaluation R2 when the group is shuffled "
          "(median, min, max over floors) ===")
    print(gsum.to_string())
    print(f"\nSaved -> {args.tables}/explain_*.csv")


if __name__ == "__main__":
    main()
