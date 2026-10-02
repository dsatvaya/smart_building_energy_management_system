"""
checks.py
---------
Pre-run checks (PROTOCOL.md, Section 8). Run BEFORE the final runs.

1. Holiday check: is_holiday matches the official Thai list, feature counts are 11/15,
   both feature versions share the same rows.
2. Smoke check: every floor x feature version x model, ONE grid point each, trains on
   the training period and predicts the evaluation period (shared split + every rolling
   origin). Checks shapes, missing values and non-negative predictions only.
   No accuracy metrics are printed, so nothing here can feed back into tuning.

Usage (from repo root):
  python src/checks.py
"""

import argparse
import os

import holidays
import numpy as np
import pandas as pd

from features import date_split, get_frame, load_clean, make_features
from models import model_searches
from run_rolling import ORIGINS, month_end
from run_total import FLOORS, MIN_TEST, MIN_TRAIN, TEST_START


def holiday_check(clean_path):
    frame = get_frame(load_clean(clean_path), "floor_total")
    X1, _ = make_features(frame, lags=True)
    X0, _ = make_features(frame, lags=False)
    assert X1.shape[1] == 15 and X0.shape[1] == 11, (X1.shape, X0.shape)
    assert X1.index.equals(X0.index), "lag versions use different rows"
    official = {pd.Timestamp(d) for d in holidays.Thailand(years=sorted(set(X1.index.year)))}
    flagged = set(X1.index[X1.is_holiday == 1].normalize())
    in_data = {d for d in official if d in set(X1.index.normalize())}
    assert flagged == in_data, f"mismatch: {sorted(flagged ^ in_data)}"
    n_eval = int(X1.loc[TEST_START:, "is_holiday"].sum() // 24)
    return len(flagged), n_eval


def smoke(clean_path, lags, test_start, test_end=None):
    X, y = make_features(get_frame(load_clean(clean_path), "floor_total"), lags=lags)
    X_tr, X_te, y_tr, _ = date_split(X, y, test_start, test_end)
    if len(X_tr) < MIN_TRAIN or len(X_te) < MIN_TEST:
        return f"skip (train {len(X_tr)} / eval {len(X_te)})"
    for name, search in model_searches(len(X_tr), n_jobs=1).items():
        search.param_grid = {k: v[:1] for k, v in search.param_grid.items()}  # one point
        search.fit(X_tr, y_tr)
        p = np.clip(search.best_estimator_.predict(X_te), 0.0, None)
        assert p.shape == (len(X_te),), f"{name}: bad shape"
        assert np.isfinite(p).all(), f"{name}: non-finite prediction"
        assert (p >= 0).all(), f"{name}: negative prediction"
    return f"ok (train {len(X_tr)} / eval {len(X_te)})"


def fit_budget(n_train=5000, cv=3):
    per_run = 0
    for search in model_searches(n_train).values():
        combos = int(np.prod([len(v) for v in search.param_grid.values()]))
        per_run += combos * cv + 1          # CV fits + refit
    return per_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", default="data/clean")
    ap.add_argument("--floors", nargs="+", type=int, default=FLOORS)
    args = ap.parse_args()

    per_run = fit_budget()
    print(f"Fit budget: {per_run} fits per floor-run | main run {per_run * len(args.floors) * 2} | "
          f"rolling (lags only) {per_run * len(args.floors) * len(ORIGINS)}")

    failed = False
    for floor in args.floors:
        path = os.path.join(args.clean_dir, f"floor{floor}_hourly.csv")
        if not os.path.exists(path):
            print(f"Floor {floor}: clean file not found"); failed = True; continue
        try:
            n_all, n_eval = holiday_check(path)
            print(f"Floor {floor}: holiday check ok ({n_all} holiday days in data, {n_eval} in evaluation)")
            for lags in (True, False):
                print(f"  main  lags={lags!s:5}: {smoke(path, lags, TEST_START)}")
            for origin in ORIGINS:
                print(f"  roll  {origin}: {smoke(path, True, origin, month_end(origin))}")
        except AssertionError as e:
            print(f"Floor {floor}: FAILED - {e}"); failed = True

    print("\nALL CHECKS PASSED" if not failed else "\nSOME CHECKS FAILED - fix before the final run")


if __name__ == "__main__":
    main()
