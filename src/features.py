"""
features.py
-----------
Clean hourly CSV (from cleaning.py) -> X, y for one target.

  target = "floor_total"  -> experiment 1 (uses floor-average sensors)
  target = "z1_ac" etc.   -> experiment 2 (uses that zone's sensors,
                             falls back to floor average if the zone has none)

lags=True  : sensors + calendar (incl. Thai public holiday flag) + lag_1, lag_24, lag_168, roll_24
lags=False : sensors + calendar only
Both versions return the SAME rows, so their scores are directly comparable.

Usage (quick check):
  python src/features.py --clean data/clean/floor7_hourly.csv --target floor_total
"""

import argparse

import holidays
import numpy as np
import pandas as pd

SENSORS = ("temp_c", "rh_pct", "lux")
LAGS = (1, 24, 168)
ROLL = 24


def load_clean(path):
    return pd.read_csv(path, parse_dates=["Date"], index_col="Date").asfreq("h")


def get_frame(clean, target):
    """Target column + the matching sensor columns."""
    if target not in clean.columns:
        raise ValueError(f"'{target}' not in file. Options: {list(clean.columns)}")
    frame = clean[[target]].rename(columns={target: "target"})
    zone = target.split("_")[0] if target.startswith("z") else None
    for s in SENSORS:
        zcol = f"{zone}_{s}" if zone else None
        frame[s] = clean[zcol] if zcol in clean.columns else clean[s]
    return frame


def make_features(frame, lags=True):
    """Calendar + sensor (+ lag) features. Returns X, y."""
    f = frame.copy()
    f["hour"] = f.index.hour
    f["dow"] = f.index.dayofweek
    f["is_weekend"] = (f["dow"] >= 5).astype(int)
    th = holidays.Thailand(years=sorted(set(f.index.year)))   # official Thai public holidays
    f["is_holiday"] = f.index.normalize().isin(pd.to_datetime(list(th.keys()))).astype(int)
    f["hour_sin"] = np.sin(2 * np.pi * f["hour"] / 24)
    f["hour_cos"] = np.cos(2 * np.pi * f["hour"] / 24)
    f["dow_sin"] = np.sin(2 * np.pi * f["dow"] / 7)
    f["dow_cos"] = np.cos(2 * np.pi * f["dow"] / 7)

    lag_cols = []
    for L in LAGS:
        f[f"lag_{L}"] = f["target"].shift(L)
        lag_cols.append(f"lag_{L}")
    f[f"roll_{ROLL}"] = f["target"].shift(1).rolling(ROLL, min_periods=ROLL // 2).mean()
    lag_cols.append(f"roll_{ROLL}")

    f = f.dropna()                       # same rows for both versions
    if not lags:
        f = f.drop(columns=lag_cols)

    y = f["target"]
    X = f.drop(columns=["target"])
    return X, y


def temporal_split(X, y, test_frac=0.2):
    """Chronological split -- never shuffle a time series."""
    cut = int(len(X) * (1 - test_frac))
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


def date_split(X, y, test_start, test_end=None):
    """Calendar split: everything before test_start trains; [test_start, test_end) tests.
    Gives every floor the same test period. test_end=None -> to the end of the data."""
    tr = X.index < pd.Timestamp(test_start)
    te = ~tr if test_end is None else (~tr) & (X.index < pd.Timestamp(test_end))
    return X[tr], X[te], y[tr], y[te]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--target", default="floor_total")
    args = ap.parse_args()

    clean = load_clean(args.clean)
    frame = get_frame(clean, args.target)
    for flag in (True, False):
        X, y = make_features(frame, lags=flag)
        X_tr, X_te, _, _ = temporal_split(X, y)
        print(f"lags={flag!s:5} | rows {len(X)} | features {X.shape[1]} | "
              f"train {X_tr.index.min():%Y-%m-%d} -> {X_tr.index.max():%Y-%m-%d} | "
              f"test {X_te.index.min():%Y-%m-%d} -> {X_te.index.max():%Y-%m-%d}")
    print("features (lags=True):", list(make_features(frame, True)[0].columns))
