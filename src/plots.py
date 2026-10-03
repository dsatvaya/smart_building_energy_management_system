"""
plots.py
--------
Figures from the FROZEN main-run predictions (evaluation period, 1 Nov - 31 Dec 2019).

  fig3_obs_vs_pred_{lags,nolags}.png  one panel per model: predicted vs actual hourly
                                      energy, all floors, perfect-fit line; title shows
                                      the model's median per-floor evaluation R2
  fig4_daily_{lags,nolags}.png        one panel per floor: daily actual vs predicted
                                      energy for that floor's CV-selected model; only days
                                      with all 24 hours present are totalled - incomplete
                                      days are left as gaps (count shown in the legend)

Training-period predictions were not saved, so only the evaluation period is plotted;
train vs evaluation is compared in the tables (train_r2 vs test_r2).

Usage (from repo root):
  python src/plots.py
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MODELS = ["Ridge", "KNN", "SVR", "RandomForest", "GradientBoosting"]


def load_preds(preds_dir, tag):
    """floor -> DataFrame (actual + one column per model) for the frozen main run."""
    out = {}
    for path in sorted(glob.glob(os.path.join(preds_dir, f"floor*_{tag}.csv"))):
        m = re.search(rf"floor(\d+)_{tag}\.csv$", os.path.basename(path))
        if m:
            out[int(m.group(1))] = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return out


def fig_obs_vs_pred(preds, allres, lags, path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    hi = max(max(p["actual"].max(), p[MODELS].max().max()) for p in preds.values()) * 1.05
    for ax, model in zip(axes.ravel(), MODELS):
        for floor, p in preds.items():
            ax.scatter(p["actual"], p[model], s=4, alpha=0.3, label=f"Floor {floor}")
        ax.plot([0, hi], [0, hi], "k--", lw=1, label="Perfect fit")
        r2 = allres.query("model == @model and lags == @lags")["test_r2"].median()
        ax.set_title(f"{model}  (median floor R\u00b2 = {r2:.3f})", fontsize=11)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.set_xlabel("Actual energy (kWh)"); ax.set_ylabel("Predicted energy (kWh)")
    axes.ravel()[-1].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes.ravel()[-1].legend(handles, labels, loc="center", markerscale=4, fontsize=11)
    version = "with lags" if lags else "without lags"
    fig.suptitle(f"Observed vs predicted hourly floor energy, evaluation period ({version})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_daily(preds, allres, lags, path):
    floors = sorted(preds)
    fig, axes = plt.subplots(len(floors), 1, figsize=(12, 2.6 * len(floors)), sharex=True)
    axes = [axes] if len(floors) == 1 else axes
    for ax, floor in zip(axes, floors):
        sel = allres.query("floor == @floor and lags == @lags and cv_rank == 1")["model"].iloc[0]
        hourly = preds[floor][["actual", sel]]
        daily = hourly.resample("D").sum(min_count=24)      # incomplete day -> NaN, not a fake dip
        n_hours = hourly["actual"].resample("D").count()
        n_bad = int((n_hours < 24).sum())
        ax.plot(daily.index, daily["actual"], "o-", ms=3,
                label=f"Actual ({n_bad} incomplete day{'s' if n_bad != 1 else ''} omitted)")
        ax.plot(daily.index, daily[sel], "s--", ms=3, label=f"Predicted ({sel})")
        ax.axvline(pd.Timestamp("2019-12-01"), color="grey", lw=0.8, ls=":")
        ax.set_ylabel(f"Floor {floor}\nkWh/day")
        ax.legend(loc="upper right", fontsize=8)
    version = "with lags" if lags else "without lags"
    axes[0].set_title(f"Daily floor energy, actual vs CV-selected model ({version}); "
                      f"dotted line = 1 Dec")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return {f: int((preds[f]["actual"].resample("D").count() < 24).sum()) for f in floors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="results/predictions")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--figures", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.figures, exist_ok=True)

    allres = pd.read_csv(os.path.join(args.tables, "total_all.csv"))   # frozen main run
    for lags, tag in ((True, "lags"), (False, "nolags")):
        preds = load_preds(args.preds, tag)
        if not preds:
            print(f"No {tag} prediction files found in {args.preds}"); continue
        fig_obs_vs_pred(preds, allres, lags, os.path.join(args.figures, f"fig3_obs_vs_pred_{tag}.png"))
        bad = fig_daily(preds, allres, lags, os.path.join(args.figures, f"fig4_daily_{tag}.png"))
        print(f"{tag}: floors {sorted(preds)} -> fig3_obs_vs_pred_{tag}.png, fig4_daily_{tag}.png")
        print(f"  incomplete days (< 24 h) omitted from daily totals: {bad}")
    print(f"Saved -> {args.figures}/")


if __name__ == "__main__":
    main()
