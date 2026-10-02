"""
cleaning.py
-----------
Raw CU-BEMS floor CSVs (1-minute) -> one clean HOURLY CSV per floor.
Model-independent: no scaling, no features. Run once.

Output columns (per floor):
  z{n}_ac, z{n}_light, z{n}_plug        hourly kWh per zone & component
  z{n}_temp_c, z{n}_rh_pct, z{n}_lux    hourly mean sensors per zone
  floor_ac, floor_light, floor_plug     summed over zones
  floor_total                           ac + light + plug (NaN if any meter missing)
  temp_c, rh_pct, lux                   floor-average sensors

Usage:
  python cleaning.py --raw data --out clean --start 2019-03-07
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

METER = re.compile(r"^z(\d+)_(AC|Li|Light|Pl|Plug)\d*(?:\(kW\))?$", re.I)
SENSOR = re.compile(r"^z(\d+)_S\d+\((lux|degC|RH%)\)$", re.I)
COMP = {"ac": "ac", "li": "light", "light": "light", "pl": "plug", "plug": "plug"}
SENS = {"lux": "lux", "degc": "temp_c", "rh%": "rh_pct"}
LIMITS = {"temp_c": (10, 45), "rh_pct": (0, 100), "lux": (0, None)}

STUCK_MIN = 1440  # same value for 24 h -> stuck (lights can sit constant all workday)
STANDBY_KW = 0.05 # meters idling at <= 0.05 kW for days (weekend standby) are real, not stuck
GAP_MIN = 60      # interpolate gaps up to 60 min
MIN_COVER = 45    # an hour needs >= 45 valid minutes


def load_floor(paths):
    """Concatenate one floor's yearly files onto a full 1-minute grid."""
    parts = []
    for p in sorted(paths):
        df = pd.read_csv(p)
        date_col = next(c for c in df.columns if c.strip().lower() in ("date", "datetime", "timestamp"))
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col)
        df.columns = [c.strip() for c in df.columns]
        parts.append(df.apply(pd.to_numeric, errors="coerce"))
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    full = pd.date_range(df.index.min().floor("min"), df.index.max().floor("min"), freq="min")
    return df.reindex(full)


def classify(columns):
    """Map raw column -> (zone, kind, clean_name)."""
    out = {}
    for c in columns:
        m = METER.match(c)
        if m:
            out[c] = (int(m.group(1)), "meter", COMP[m.group(2).lower()])
            continue
        m = SENSOR.match(c)
        if m:
            out[c] = (int(m.group(1)), "sensor", SENS[m.group(2).lower()])
    return out


def mask_stuck(s, floor_value):
    run_id = s.ne(s.shift()).cumsum()
    run_len = s.groupby(run_id).transform("size")
    return s.mask((run_len >= STUCK_MIN) & (s > floor_value) & s.notna())


def clean_minutes(df, cols):
    df = df[list(cols)].copy()
    for c, (_, kind, name) in cols.items():
        s = df[c]
        if kind == "meter":
            s = s.mask(s < 0)
        else:
            lo, hi = LIMITS[name]
            s = s.mask(s < lo) if lo is not None else s
            s = s.mask(s > hi) if hi is not None else s
        s = mask_stuck(s, STANDBY_KW if kind == "meter" else float("-inf"))
        df[c] = s.interpolate(limit=GAP_MIN, limit_area="inside")
    return df


def to_hourly(df):
    mean = df.resample("h").mean()
    cnt = df.resample("h").count()
    return mean.mask(cnt < MIN_COVER)


def aggregate(hourly, cols):
    out = pd.DataFrame(index=hourly.index)
    groups = {}
    for c, (z, kind, name) in cols.items():
        groups.setdefault((z, kind, name), []).append(c)

    for (z, kind, name), cs in sorted(groups.items()):
        block = hourly[cs]
        # meters: sum units (NaN if any unit missing); sensors: average
        out[f"z{z}_{name}"] = block.sum(axis=1, min_count=len(cs)) if kind == "meter" else block.mean(axis=1)

    for comp in ("ac", "light", "plug"):
        zc = [c for c in out.columns if c.endswith(f"_{comp}")]
        if zc:
            out[f"floor_{comp}"] = out[zc].sum(axis=1, min_count=len(zc))
    fc = [c for c in ("floor_ac", "floor_light", "floor_plug") if c in out]
    out["floor_total"] = out[fc].sum(axis=1, min_count=len(fc))

    for name in ("temp_c", "rh_pct", "lux"):
        zc = [c for c in out.columns if re.fullmatch(rf"z\d+_{name}", c)]
        if zc:
            out[name] = out[zc].mean(axis=1)
    return out


def missing_report(raw, cols, floor):
    rows = [{"floor": floor, "column": c, "zone": z, "type": name,
             "missing_pct": round(100 * raw[c].isna().mean(), 2)}
            for c, (z, _, name) in cols.items()]
    return pd.DataFrame(rows)


def checks(out, floor, raw_missing_pct):
    kwh = [c for c in out.columns if c.endswith(("_ac", "_light", "_plug")) or c == "floor_total"]
    assert (out[kwh].fillna(0) >= 0).all().all(), f"{floor}: negative kWh"
    ok = out["floor_total"].notna()
    parts = out.loc[ok, [c for c in ("floor_ac", "floor_light", "floor_plug") if c in out]].sum(axis=1)
    assert np.allclose(parts, out.loc[ok, "floor_total"]), f"{floor}: total != sum of parts"
    hours_kept = 100 * ok.mean()
    flag = "OK" if raw_missing_pct < 15 else "ABOVE 15% (paper threshold)"
    print(f"  raw missing {raw_missing_pct:.2f}% [{flag}] | usable floor_total hours {hours_kept:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data")
    ap.add_argument("--out", default="clean")
    ap.add_argument("--start", default=None,
                    help="drop data before this date, e.g. 2019-03-07 (sensors start 6 Mar 2019)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = {}
    for p in glob.glob(os.path.join(args.raw, "**", "*Floor*.csv"), recursive=True):
        m = re.search(r"(\d{4})Floor(\d+)", os.path.basename(p), re.I)
        if m:
            files.setdefault(int(m.group(2)), []).append(p)
    if not files:
        raise SystemExit(f"No *FloorN.csv files under {args.raw}")

    reports = []
    for floor, paths in sorted(files.items()):
        print(f"Floor {floor}: {[os.path.basename(p) for p in sorted(paths)]}")
        raw = load_floor(paths)
        if args.start:
            raw = raw.loc[args.start:]
        cols = classify(raw.columns)
        reports.append(missing_report(raw, cols, floor))
        raw_missing = 100 * raw[list(cols)].isna().to_numpy().mean()

        out = aggregate(to_hourly(clean_minutes(raw, cols)), cols)
        checks(out, floor, raw_missing)
        out.index.name = "Date"
        out.to_csv(os.path.join(args.out, f"floor{floor}_hourly.csv"))

    pd.concat(reports).to_csv(os.path.join(args.out, "missing_report.csv"), index=False)
    print(f"Done -> {args.out}/floorN_hourly.csv, missing_report.csv")


if __name__ == "__main__":
    main()