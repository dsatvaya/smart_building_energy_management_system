"""Leakage-aware model comparison for CU-BEMS hourly energy forecasting.

Run from the repository root, for example:
    python src/models.py --clean data/clean/floor7_hourly.csv --target z1_ac

Each forecast is one hour ahead and uses the observed lagged consumption values
provided by features.py. The test period is held out chronologically; tuning uses
only expanding-window TimeSeriesSplit folds from the training period.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:  # Support both `python src/models.py` and package-style imports.
    from .features import date_split, get_frame, load_clean, make_features, temporal_split
    from .metrics import regression_metrics
except ImportError:
    from features import date_split, get_frame, load_clean, make_features, temporal_split
    from metrics import regression_metrics


def clipped_neg_rmse(estimator, X, y):
    """CV scorer on the SAME clipped predictions we report on test (energy >= 0)."""
    pred = np.clip(estimator.predict(X), 0.0, None)
    return -float(np.sqrt(np.mean((np.asarray(y) - pred) ** 2)))


def model_searches(n_train, cv_splits=3, random_state=42, n_jobs=-1):
    """Build model-specific preprocessing pipelines and tuning grids."""
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2")
    if n_train < cv_splits + 1:
        raise ValueError(
            f"Need at least {cv_splits + 1} training rows for {cv_splits} time-series folds; "
            f"got {n_train}."
        )

    cv = TimeSeriesSplit(n_splits=cv_splits)
    smallest_fold_train = n_train // (cv_splits + 1)
    neighbors = [k for k in (3, 5, 9, 15, 25, 40) if k <= smallest_fold_train]
    if not neighbors:
        neighbors = [1]

    scaled_prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    tree_prep = SimpleImputer(strategy="median")

    specs = {
        "Ridge": (
            Pipeline([("prep", scaled_prep), ("model", Ridge())]),
            {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
        ),
        "KNN": (
            Pipeline([("prep", scaled_prep), ("model", KNeighborsRegressor(n_jobs=1))]),
            {"model__n_neighbors": neighbors,
             "model__weights": ["uniform", "distance"],
             "model__p": [1, 2]},
        ),
        "SVR": (
            Pipeline([("prep", scaled_prep), ("model", SVR(kernel="rbf"))]),
            {"model__C": [1.0, 10.0, 100.0, 1000.0],
             "model__epsilon": [0.1, 0.5, 1.0],
             "model__gamma": ["scale", 0.01, 0.1]},
        ),
        "RandomForest": (
            Pipeline([("prep", tree_prep),
                     ("model", RandomForestRegressor(random_state=random_state, n_jobs=1))]),
            {"model__n_estimators": [300],
             "model__max_depth": [None, 12, 20],
             "model__min_samples_leaf": [1, 3, 5],
             "model__max_features": [1.0, 0.5]},
        ),
        "GradientBoosting": (
            Pipeline([("prep", tree_prep),
                     ("model", GradientBoostingRegressor(random_state=random_state))]),
            {"model__n_estimators": [200, 500],
             "model__learning_rate": [0.03, 0.1],
             "model__max_depth": [2, 3, 5],
             "model__subsample": [0.8, 1.0]},
        ),
    }

    searches = {}
    for name, (pipeline, grid) in specs.items():
        searches[name] = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring=clipped_neg_rmse,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
            error_score="raise",
            return_train_score=False,
        )
    return searches


def compare_models(X_train, y_train, X_test, y_test, cv_splits=3,
                   random_state=42, n_jobs=-1):
    """Tune on training history, evaluate each refitted model on the test tail.

    Returns a score-sorted DataFrame and a mapping of names to fitted searches.
    Do not use the test results to choose hyperparameters or repeatedly revise
    the model; use training-only CV for selection and reserve the test tail for
    the final comparison.
    """
    searches = model_searches(len(X_train), cv_splits, random_state, n_jobs)
    rows = []
    fitted = {}
    for name, search in searches.items():
        search.fit(X_train, y_train)
        fitted[name] = search
        cv_rmse = -search.best_score_
        cv_std = search.cv_results_["std_test_score"][search.best_index_]
        # energy can't be negative: clip linear-model undershoot on low-load hours
        prediction = np.clip(search.best_estimator_.predict(X_test), 0.0, None)
        train_pred = np.clip(search.best_estimator_.predict(X_train), 0.0, None)
        scores = regression_metrics(y_test, prediction)
        train_scores = regression_metrics(y_train, train_pred)
        row = {
            "model": name,
            "cv_rmse_mean": float(cv_rmse),
            "cv_rmse_std": float(cv_std),
            # scale-free: CV RMSE as % of mean training load, comparable across floors
            "cv_cvrmse_pct": float(cv_rmse / np.mean(y_train) * 100),
            "best_params": search.best_params_,
        }
        row.update({f"train_{key}": value for key, value in train_scores.items()})
        row.update({f"test_{key}": value for key, value in scores.items()})
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("cv_rmse_mean").reset_index(drop=True)
    return results, fitted


def run_experiment(clean_path, target="floor_total", use_lags=True,
                   test_frac=0.2, cv_splits=3, random_state=42, n_jobs=-1,
                   test_start=None):
    """Load a cleaned floor file, create features, split chronologically, compare.
    test_start (e.g. "2019-11-01") gives a fixed calendar test period; otherwise
    the last test_frac of rows is used."""
    clean = load_clean(clean_path)
    frame = get_frame(clean, target)
    X, y = make_features(frame, lags=use_lags)
    if test_start:
        X_train, X_test, y_train, y_test = date_split(X, y, test_start)
    else:
        X_train, X_test, y_train, y_test = temporal_split(X, y, test_frac=test_frac)
    if X_train.empty or X_test.empty:
        raise ValueError("Chronological split produced an empty train or test set")

    results, fitted = compare_models(
        X_train, y_train, X_test, y_test, cv_splits, random_state, n_jobs
    )
    metadata = {
        "target": target,
        "lags": use_lags,
        "n_features": X.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_start": str(X_train.index.min()),
        "train_end": str(X_train.index.max()),
        "test_start": str(X_test.index.min()),
        "test_end": str(X_test.index.max()),
    }
    for key, value in metadata.items():
        results[key] = value
    return results, fitted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", required=True, help="Clean hourly CSV from cleaning.py")
    parser.add_argument("--target", default="floor_total", help="e.g. floor_total, z1_ac")
    parser.add_argument("--no-lags", action="store_true", help="Use calendar and sensor features only")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--test-start", default=None, help="fixed test start date, e.g. 2019-11-01")
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--out", default=None, help="Optional CSV path for comparison results")
    args = parser.parse_args()

    if not 0 < args.test_frac < 1:
        parser.error("--test-frac must be between 0 and 1")
    results, _ = run_experiment(
        args.clean, args.target, not args.no_lags, args.test_frac,
        args.cv_splits, args.random_state, args.n_jobs, args.test_start,
    )
    display = ["model", "cv_rmse_mean", "cv_rmse_std", "test_rmse",
               "test_r2", "test_mape_nonzero_pct", "test_smape_pct",
               "test_cvrmse_pct", "test_nmbe_pct", "test_total_energy_abs_error_pct"]
    print(results[display].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSelection rule: rank by training-only TimeSeriesSplit CV RMSE; inspect all test metrics")
    print("before finalizing. Test metrics are not used for tuning.")

    if args.out:
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        results.to_csv(out_path, index=False)
        print(f"\nSaved comparison table: {out_path}")


if __name__ == "__main__":
    main()
