# RP: Interpretable ML for Smart Building Energy Management — Implementation Plan

## The one-sentence pitch

> Sari et al. (2023) predicted building energy use with k-NN and got MAPE 22.01%, with no explanation of *why* the model predicts what it predicts. Gugliermetti et al. (2024) argue that this missing explanation is the main barrier to real adoption. We reproduce that baseline on the same public dataset, replace k-NN with interpretable ensembles, and attach SHAP so a facility manager can see which factor drove each prediction — then adapt it to a hot-humid Indian climate.

That is a defensible research contribution, not just a re-implementation. Lead with it.

## Why this framing works

The three papers hand you the gap almost explicitly:

| Paper | What it gives you |
|---|---|
| Sari et al. (2023), ITcon | The dataset (CU-BEMS) and the number to beat: **MAPE 22.01%, avg relative error 17.76%**. Explicitly asks future work to compare other algorithms. |
| Pokkuluri et al. (2025), JISEM | Evidence that **RF (test R² 0.84) beats ANN (test R² 0.35, badly overfit)**, and that HVAC (~41%) + lighting (~27%) dominate the variance. |
| Gugliermetti et al. (2024), Energies | The interpretability argument, the SHAP/LIME toolkit, and the accuracy–interpretability tradeoff figure. |

So: better model + explanations + local adaptation. Each piece is cited, none is hand-wavy.

## Step 1 — Get the data (tonight, ~30 min)

CU-BEMS: seven floors of Chamchuri 5, 1-minute readings, July 2018 – Dec 2019.

Easiest source — Kaggle: `claytonmiller/cubems-smart-building-energy-and-iaq-data`. Files are `2018Floor1.csv` … `2019Floor7.csv`, ~733 MB total. Original citation is Pipattanasomporn et al. (2020), *Scientific Data* 7:1.

Grab `2019Floor6.csv` first; it has the lowest missing rate and the paper's best MAPE (15.37%), so it's the cleanest place to start.

Schema notes (confirmed): meter columns carry a unit suffix — `z1_Light(kW)`, `z1_Plug(kW)`, `z2_AC1(kW)`. Sensors are `z1_S1(degC)`, `z1_S1(RH%)`, `z1_S1(lux)`. **Floor 1 has no environmental sensors** — only floors 2–7 do, so don't pick Floor 1 for any model that uses temperature or humidity.

Put it in `data/`.

## Step 2 — Run the pipeline (tomorrow morning)

```bash
pip install pandas numpy scikit-learn matplotlib shap

# smoke test, no data needed
python bems_pipeline.py --synthetic --zone 1 --component light

# real run
python bems_pipeline.py --csv data/2019Floor6.csv --zone 1 --component light --shap
```

It produces, in `results/`:
- `missing_report.csv` — your version of the paper's Table 2
- `metrics_zone1_light.csv` — Ridge vs k-NN vs Random Forest, head to head
- `pred_vs_actual_*.png` — your version of the paper's Fig. 7
- `rf_feature_importance.csv`, `shap_summary.png`, `shap_global_importance.csv`

Three of those four figures map onto figures in the paper you reviewed. That's what "we've been working on the code" looks like with evidence behind it.

## Step 3 — Design choices to defend in the meeting

These are the things a supervisor will probe. Have answers ready:

1. **Chronological split, not random.** Random `train_test_split` on a time series leaks the future into training and inflates R². The paper used Jan–Nov train / Dec test; we do the same with a trailing 20%.
2. **Lag features.** The paper's k-NN used only hour and day one-hots. Adding lag-1, lag-24, lag-168 and a 24-hour rolling mean encodes thermal inertia — Li et al. (2021) found past consumption dominates cooling-load prediction. This alone should move MAPE substantially.
3. **MAPE is fragile here.** Night-time loads approach zero, so percentage error blows up. We report MAPE (on non-zero hours, for comparability with the paper), plus sMAPE, RMSE and R², plus total-energy relative error.
4. **Cyclical hour encoding** instead of 24 one-hot columns — kNN is distance-based, and 31 sparse binary columns dilute the distance metric.

## Step 4 — Weeks ahead (the "clear line of action")

| Week | Deliverable |
|---|---|
| 1 | Pipeline runs on all 7 floors × 3 components. Baseline comparison table complete. |
| 2 | Add Gradient Boosting / XGBoost. Hyperparameter tuning with `TimeSeriesSplit`. Full metrics matrix vs the paper's Table 6. |
| 3 | SHAP: global importance + local waterfall plots for specific anomalous hours. Compare against RF built-in importance (they disagree in interesting ways). |
| 4 | Seasonal clustering before training — Rourkela's latent cooling load behaves differently in monsoon vs dry season. This is your report's Section 5 made concrete. |
| 5 | Writeup + dashboard mock (optional; the paper's Figs. 11–13 are a template). |

## Step 5 — Handling the "what data will *we* use?" question

Your report already flagged this as the real bottleneck, so raise it before she does. Proposed answer:

- **Phase 1 (now):** CU-BEMS. Bangkok is tropical and hot-humid — genuinely close to Odisha. Method development happens here.
- **Phase 2:** Validate on NIT-R campus data. Ask her directly tomorrow whether sub-metered feeds exist for any academic block, or whether a small IoT node (DHT22 + a clamp meter on a lab feeder) is feasible. Even 4–6 weeks of one-lab data makes the transfer-learning story real.
- **Fallback:** EnergyPlus digital twin calibrated to Rourkela TMY weather, as your report proposed.

Asking this question makes you look like you're planning ahead, and it gets you an answer you actually need.

## One honest note

Don't present synthetic-mode output as results. Run it on the real CSV before the meeting — it takes minutes once downloaded. If something breaks and you only have the smoke test, say "pipeline is working end-to-end, validating on CU-BEMS this week" and show the code. That's true and it's enough. A fabricated MAPE that she later can't reproduce is a far worse position than a pipeline with no numbers yet.
