"""
bems_pipeline.py
----------------
End-to-end pipeline for hourly building energy-use prediction on the CU-BEMS
dataset (Chamchuri 5, Bangkok) -- the same public data used by Sari et al. (2023).

What it does:
  1. Reads a raw floor CSV (1-minute resolution).
  2. Reports missing data per zone/component (reproduces the paper's Table 2).
  3. Resamples 1-min kW readings to hourly kWh, per zone and component.
  4. Builds features: hour-of-day, day-of-week, sensor readings (lux/temp/RH),
     lag and rolling-mean features.
  5. Trains and compares: Ridge (linear floor), k-NN (paper's baseline), Random Forest.
  6. Reports MAPE / sMAPE / RMSE / R^2 / total-energy relative error.
  7. Optional SHAP explanation of the Random Forest.

Usage:
  # smoke test with generated data (no download needed)
  python bems_pipeline.py --synthetic --zone 1 --component light

  # real data
  python bems_pipeline.py --csv data/2019Floor6.csv --zone 1 --component light --shap

Data source: https://github.com/nasrin-sultana/CU-BEMS  (or the SGRU portal linked
in Sari et al.). Files are named like 2018Floor1.csv ... 2019Floor7.csv.
"""

import argparse
import os
import re
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score


# --------------------------------------------------------------------------
# 1. Column parsing
# --------------------------------------------------------------------------
# CU-BEMS columns look like: z1_AC1, z1_Light, z1_Plug, z1_S1(lux),
# z1_S1(degC), z1_S1(RH%).  Naming varies slightly between releases, so we
# match with regex instead of hardcoding.

COMPONENT_PATTERNS = {
    "ac":    re.compile(r"^z(\d+)_AC\d*(?:\(kW\))?$", re.I),
    "light": re.compile(r"^z(\d+)_(?:Li|Light)\d*(?:\(kW\))?$", re.I),
    "plug":  re.compile(r"^z(\d+)_(?:Pl|Plug)\d*(?:\(kW\))?$", re.I),
}
SENSOR_PATTERN = re.compile(r"^z(\d+)_S\d+\((lux|degC|RH%)\)$", re.I)

SENSOR_ALIAS = {"lux": "lux", "degc": "temp_c", "rh%": "rh_pct"}


def component_columns(columns, zone, component):
    """Return all meter columns for a given zone + component."""
    pat = COMPONENT_PATTERNS[component]
    out = []
    for c in columns:
        m = pat.match(c.strip())
        if m and int(m.group(1)) == zone:
            out.append(c)
    return out


def sensor_columns(columns, zone):
    """Return {clean_name: [raw columns]} for a zone's sensors."""
    out = {}
    for c in columns:
        m = SENSOR_PATTERN.match(c.strip())
        if m and int(m.group(1)) == zone:
            key = SENSOR_ALIAS[m.group(2).lower()]
            out.setdefault(key, []).append(c)
    return out


def available_zones(columns):
    zones = set()
    for c in columns:
        for pat in list(COMPONENT_PATTERNS.values()) + [SENSOR_PATTERN]:
            m = pat.match(c.strip())
            if m:
                zones.add(int(m.group(1)))
    return sorted(zones)


# --------------------------------------------------------------------------
# 2. Loading + missing-data report
# --------------------------------------------------------------------------

def load_floor(path):
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if c.strip().lower() in ("date", "datetime", "timestamp")), None)
    if date_col is None:
        raise ValueError(f"No date column found. Columns: {list(df.columns)[:10]}")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    return df


def missing_report(df):
    """Reproduces the paper's missing-data table, per column."""
    expected = len(df)
    rows = []
    for c in df.columns:
        n_missing = int(df[c].isna().sum())
        rows.append({
            "column": c,
            "missing": n_missing,
            "expected": expected,
            "missing_pct": round(100.0 * n_missing / expected, 2) if expected else np.nan,
        })
    rep = pd.DataFrame(rows).sort_values("missing_pct", ascending=False)
    overall = round(100.0 * df.isna().sum().sum() / (expected * df.shape[1]), 2)
    return rep, overall


# --------------------------------------------------------------------------
# 3. Resample to hourly kWh + assemble the modelling frame
# --------------------------------------------------------------------------

def build_frame(df, zone, component, max_gap_minutes=60):
    """
    Returns an hourly DataFrame with:
      target   -- kWh consumed by <component> in <zone> that hour
      lux / temp_c / rh_pct -- hourly mean sensor readings for that zone
    Mean of kW over an hour == kWh for that hour, so .resample('h').mean() is correct.
    """
    e_cols = component_columns(df.columns, zone, component)
    if not e_cols:
        raise ValueError(
            f"No '{component}' meters found for zone {zone}. "
            f"Zones present: {available_zones(df.columns)}"
        )

    energy = df[e_cols].apply(pd.to_numeric, errors="coerce")
    # short sensor dropouts are interpolated; long gaps stay NaN and get dropped
    energy = energy.interpolate(limit=max_gap_minutes, limit_area="inside")
    target = energy.sum(axis=1, min_count=1).resample("h").mean().rename("target")

    frame = target.to_frame()

    for name, cols in sensor_columns(df.columns, zone).items():
        s = df[cols].apply(pd.to_numeric, errors="coerce")
        s = s.interpolate(limit=max_gap_minutes, limit_area="inside")
        frame[name] = s.mean(axis=1).resample("h").mean()

    return frame


def make_features(frame, lags=(1, 24, 168), roll=24):
    """Calendar + sensor + autoregressive features. Returns X, y."""
    f = frame.copy()

    f["hour"] = f.index.hour
    f["dow"] = f.index.dayofweek
    f["is_weekend"] = (f["dow"] >= 5).astype(int)

    # cyclical encoding is cheaper than 31 one-hot columns and works better for trees+kNN
    f["hour_sin"] = np.sin(2 * np.pi * f["hour"] / 24)
    f["hour_cos"] = np.cos(2 * np.pi * f["hour"] / 24)
    f["dow_sin"] = np.sin(2 * np.pi * f["dow"] / 7)
    f["dow_cos"] = np.cos(2 * np.pi * f["dow"] / 7)

    for L in lags:
        f[f"lag_{L}"] = f["target"].shift(L)
    f[f"roll_{roll}"] = f["target"].shift(1).rolling(roll, min_periods=roll // 2).mean()

    f = f.dropna()
    y = f["target"]
    X = f.drop(columns=["target"])
    return X, y


# --------------------------------------------------------------------------
# 4. Metrics
# --------------------------------------------------------------------------

def evaluate(y_true, y_pred, eps=1e-3):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.abs(y_true) > eps  # MAPE explodes on near-zero night-time loads
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else np.nan

    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = float(np.mean(np.abs(y_true - y_pred) / np.where(denom > eps, denom, np.nan)) * 100)

    total_true, total_pred = y_true.sum(), y_pred.sum()
    rel_err_total = float(abs(total_pred - total_true) / total_true * 100) if total_true else np.nan

    return {
        "MAPE_%": round(mape, 2),
        "sMAPE_%": round(smape, 2),
        "RMSE_kWh": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "TotalEnergyRelErr_%": round(rel_err_total, 2),
    }


# --------------------------------------------------------------------------
# 5. Models
# --------------------------------------------------------------------------

def build_models(n_train):
    """k-NN reproduces Sari et al.; RF is the interpretable challenger."""
    cv = TimeSeriesSplit(n_splits=min(4, max(2, n_train // 500)))

    knn = GridSearchCV(
        Pipeline([("scale", MinMaxScaler()), ("model", KNeighborsRegressor())]),
        {"model__n_neighbors": [3, 5, 7, 10, 15, 20],
         "model__weights": ["uniform", "distance"]},
        scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1,
    )

    ridge = Pipeline([("scale", MinMaxScaler()), ("model", Ridge(alpha=1.0))])

    rf = RandomForestRegressor(
        n_estimators=300, max_depth=30, min_samples_split=2,
        random_state=6, n_jobs=-1,
    )

    return {"Ridge (linear floor)": ridge, "kNN (paper baseline)": knn, "RandomForest": rf}


def temporal_split(X, y, test_frac=0.2):
    """Chronological split -- never shuffle a time series."""
    cut = int(len(X) * (1 - test_frac))
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


# --------------------------------------------------------------------------
# 6. Synthetic data (smoke test only -- NOT for reporting results)
# --------------------------------------------------------------------------

def synthetic_floor_csv(path, days=90, seed=0):
    """Generates a CU-BEMS-shaped 1-minute CSV so the pipeline runs without the download."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=days * 24 * 60, freq="min")
    hour = idx.hour.values
    dow = idx.dayofweek.values

    occupied = ((hour >= 8) & (hour <= 18) & (dow < 5)).astype(float)
    season = np.sin(2 * np.pi * idx.dayofyear.values / 365)

    light = occupied * 3.0 + 0.1 + rng.normal(0, 0.15, len(idx))
    ac1 = occupied * (18 + 6 * season) + 1.0 + rng.normal(0, 1.5, len(idx))
    ac2 = occupied * (12 + 4 * season) + 0.8 + rng.normal(0, 1.2, len(idx))
    plug = occupied * 5.0 + 1.2 + rng.normal(0, 0.3, len(idx))

    lux = occupied * 55 + rng.normal(0, 4, len(idx))
    temp = 24 + 2 * season + occupied * 1.2 + rng.normal(0, 0.4, len(idx))
    rh = 55 + 8 * season - occupied * 4 + rng.normal(0, 2, len(idx))

    df = pd.DataFrame({
        "Date": idx,
        "z1_Light(kW)": np.clip(light, 0, None),
        "z1_Plug(kW)": np.clip(plug, 0, None),
        "z1_AC1(kW)": np.clip(ac1, 0, None),
        "z1_AC2(kW)": np.clip(ac2, 0, None),
        "z1_S1(lux)": np.clip(lux, 0, None),
        "z1_S1(degC)": temp,
        "z1_S1(RH%)": np.clip(rh, 0, 100),
    })
    # inject realistic sensor dropouts
    drop = rng.random(len(df)) < 0.03
    df.loc[drop, ["z1_AC1(kW)", "z1_S1(lux)"]] = np.nan

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------
# 7. Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="path to a CU-BEMS floor CSV (e.g. 2019Floor6.csv)")
    ap.add_argument("--synthetic", action="store_true", help="generate fake data and run a smoke test")
    ap.add_argument("--zone", type=int, default=1)
    ap.add_argument("--component", choices=["ac", "light", "plug"], default="light")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--shap", action="store_true", help="compute SHAP values for the Random Forest")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.synthetic:
        path = synthetic_floor_csv(os.path.join(args.outdir, "_synthetic_floor.csv"))
        print("[!] SYNTHETIC MODE -- numbers below are a pipeline smoke test, not results.\n")
    elif args.csv:
        path = args.csv
    else:
        ap.error("pass --csv <file> or --synthetic")

    df = load_floor(path)
    print(f"Loaded {path}: {len(df):,} rows, {df.shape[1]} columns")
    print(f"Range: {df.index.min()} -> {df.index.max()}")
    print(f"Zones detected: {available_zones(df.columns)}\n")

    rep, overall = missing_report(df)
    rep.to_csv(os.path.join(args.outdir, "missing_report.csv"), index=False)
    print(f"Overall missing data: {overall}%  (paper's acceptance threshold: <15%)")
    print(rep.head(8).to_string(index=False), "\n")

    frame = build_frame(df, args.zone, args.component)
    X, y = make_features(frame)
    print(f"Modelling frame: {len(X):,} hourly samples, {X.shape[1]} features")
    print(f"Features: {list(X.columns)}\n")

    X_tr, X_te, y_tr, y_te = temporal_split(X, y, args.test_frac)
    print(f"Train {len(X_tr):,} ({X_tr.index.min().date()} -> {X_tr.index.max().date()})  |  "
          f"Test {len(X_te):,} ({X_te.index.min().date()} -> {X_te.index.max().date()})\n")

    results, fitted = {}, {}
    for name, model in build_models(len(X_tr)).items():
        model.fit(X_tr, y_tr)
        results[name] = evaluate(y_te, model.predict(X_te))
        fitted[name] = model
        if hasattr(model, "best_params_"):
            results[name]["best_params"] = model.best_params_
        print(f"{name:24s} {results[name]}")

    table = pd.DataFrame(results).T
    table.to_csv(os.path.join(args.outdir, f"metrics_zone{args.zone}_{args.component}.csv"))
    with open(os.path.join(args.outdir, "metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    rf = fitted["RandomForest"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    imp.to_csv(os.path.join(args.outdir, "rf_feature_importance.csv"))
    print("\nTop RF features:")
    print(imp.head(10).to_string())

    _plot_predictions(X_te, y_te, fitted, args)

    if args.shap:
        _run_shap(rf, X_te, args.outdir)

    print(f"\nArtifacts written to ./{args.outdir}/")


def _plot_predictions(X_te, y_te, fitted, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[skip] matplotlib not installed")
        return

    n = min(len(y_te), 24 * 14)          # first two test weeks
    window = y_te.iloc[:n]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(window.index, window.values, label="Actual", lw=1.6, color="black")
    for name, model in fitted.items():
        pred = model.predict(X_te.iloc[:n])
        ax.plot(window.index, pred, label=name, lw=1.1, alpha=0.85)

    ax.set_ylabel("kWh")
    ax.set_title(f"Zone {args.zone} - {args.component} (first {n // 24} test days)")
    ax.legend()
    fig.tight_layout()

    out = os.path.join(args.outdir, f"pred_vs_actual_zone{args.zone}_{args.component}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nSaved plot: {out}")


def _run_shap(rf, X_te, outdir):
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[skip] `pip install shap` to enable explanations")
        return

    sample = X_te.sample(min(500, len(X_te)), random_state=0)
    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(sample)

    shap.summary_plot(sv, sample, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "shap_summary.png"), dpi=130)
    plt.close()

    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=sample.columns).sort_values(ascending=False)
    mean_abs.to_csv(os.path.join(outdir, "shap_global_importance.csv"))
    print("\nSHAP global importance (mean |SHAP|):")
    print(mean_abs.head(10).to_string())


if __name__ == "__main__":
    main()
