# Experiment Protocol (frozen)

Committed **before** the final run. Nothing below is changed after results are seen.
Any later deviation is listed under *Amendments* with its reason.

## Disclosure
The 1 Nov – 31 Dec 2019 period was already evaluated in an earlier run. Two decisions
below were made after seeing those results: the **expanded grids** and the **holiday flag**.
That period is therefore reported as an **evaluation period**, not an untouched test set.
A full rolling-origin evaluation was considered and left to future work (Section 7).

## 1. Data
- Source: CU-BEMS (Pipattanasomporn et al., 2020, *Scientific Data*), Kaggle mirror
  `claytonmiller/cubems-smart-building-energy-and-iaq-data`.
- Period used: **7 Mar 2019 – 31 Dec 2019**, 1-minute data aggregated to hourly.
- 2018 excluded: indoor sensors offline from Sep 2018 (no Oct–Dec 2018 sensor data).
- Jan – 6 Mar 2019 excluded: indoor sensors not yet recording (start 6 Mar 2019, 14:50).

## 2. Floors
| Floor | Status | Reason |
|---|---|---|
| 1 | Excluded | No indoor environmental sensors |
| 2, 3, 5, 7 | Included | Complete |
| 4 | Included | Zone 1 AC meters offline Mar–May; those hours drop out automatically |
| 6 | Excluded | Source file ends 22 Oct 2019 (checked Kaggle mirror and official repository) |

Minimum usable hours per floor: 1,600 train and 400 evaluation, otherwise the floor is skipped and logged.

## 3. Cleaning (`cleaning.py`, run with `--start 2019-03-07`)
- Out-of-range → missing: kW < 0; RH outside 0–100 %; temperature outside 10–45 °C; lux < 0.
- Stuck meter: same value for ≥ 24 h above 0.05 kW standby → missing (sensors: any value).
- Gaps ≤ 60 min filled by linear interpolation.
- Hour kept only if ≥ 45 of 60 minutes are present.
- Floor total = AC + lighting + plug over all zones; missing if any meter is missing.

## 4. Features (`features.py`)
- Target (Experiment 1): `floor_total`, hourly kWh.
- Without lags (11): `temp_c, rh_pct, lux, hour, dow, is_weekend, is_holiday,
  hour_sin, hour_cos, dow_sin, dow_cos`.
- With lags (15): the above + `lag_1, lag_24, lag_168, roll_24` (roll uses past hours only).
- `is_holiday`: official Thai public holidays from the `holidays` package (pinned version).
- Both versions use the **same rows**.

## 5. Split, tuning and scoring
- Shared calendar split for every floor: **train < 1 Nov 2019; evaluation 1 Nov – 31 Dec 2019**.
- Tuning: `GridSearchCV`, `TimeSeriesSplit(n_splits=3)` on the training period only.
- CV scorer: RMSE on predictions clipped at 0 (same clipping as reported metrics).
- Scaling (Ridge, KNN, SVR) and imputation inside each pipeline, fitted on training folds only.

### Grids
| Model | Grid | Combos |
|---|---|---|
| Ridge | alpha ∈ {0.01, 0.1, 1, 10, 100, 1000} | 6 |
| KNN | k ∈ {3, 5, 9, 15, 25, 40}; weights ∈ {uniform, distance}; p ∈ {1, 2} | 24 |
| SVR (RBF) | C ∈ {1, 10, 100, 1000}; epsilon ∈ {0.1, 0.5, 1.0}; gamma ∈ {scale, 0.01, 0.1} | 36 |
| Random Forest | 300 trees; max_depth ∈ {None, 12, 20}; min_samples_leaf ∈ {1, 3, 5}; max_features ∈ {1.0, 0.5} | 18 |
| GBM | n_estimators ∈ {200, 500}; learning_rate ∈ {0.03, 0.1}; max_depth ∈ {2, 3, 5}; subsample ∈ {0.8, 1.0} | 24 |

Random seed: 42.

## 6. Model selection and reporting
- **Selection uses training CV only.**
  - Within a floor: rank by CV RMSE.
  - Across floors: median normalised CV(RMSE) % (CV RMSE / mean training load) and mean within-floor rank.
- Evaluation-period metrics are reported as the check on generalisation, never used to choose.
- Metrics: R², MSE, RMSE, VAF, a20, IOA, CV(RMSE), NMBE (MAPE reported with caution:
  near-zero loads inflate it).
- ASHRAE Guideline 14 hourly calibration thresholds (CV(RMSE) ≤ 30 %, |NMBE| ≤ 10 %) are
  shown as a reference point only, not as a pass/fail for forecasting.

## 7. November vs December bias breakdown (reporting only)
- Computed from the saved hourly predictions in `results/predictions/` — no retraining.
- For every floor × feature set × model: R², CV(RMSE) and NMBE, separately for
  November and December 2019.
- Purpose: show whether over-prediction is concentrated in December (holiday /
  semester-break period) or present in both months.
- Reporting only: these numbers are never used to select or retune models.
- A full rolling-origin evaluation (re-tuning at several monthly origins) was considered
  and dropped for compute/time reasons; it is listed as future work.

## 8. Pre-run checks and run order
Compute budget (fits = CV fits + refit): 329 per floor-run →
**3,290 for the main run** (5 floors × 2 feature sets).

Run order, each exactly once:
1. `python src/checks.py` — holiday check (flags match the official Thai list, 11/15
   features, same rows in both versions) and smoke check (every floor × version × model,
   one grid point). Prints **no accuracy metrics**; only shapes, missing values and
   non-negative predictions. Must end with `ALL CHECKS PASSED`. A failure is a code/data
   fix only, never a grid or feature change.
   *(Its extra rolling-split smoke runs are harmless and unused.)*
   **Status: run, ALL CHECKS PASSED.**
2. `python src/run_total.py` — main run.
3. Nov vs Dec breakdown (Section 7) from the saved predictions.

No retuning after step 2: evaluation results are reported, not acted on.

## Amendments
_None yet._
