# Interpretable Machine Learning for Smart Building Energy Management

Research Project (RP), NIT Rourkela.

## Problem

Sari et al. (2023) predicted hourly energy use in a smart building using k-NN, reporting **MAPE 22.01%** and an average relative error of **17.76%** — with no explanation of *why* the model predicts what it predicts. Gugliermetti et al. (2024) argue this missing explanation is the main barrier to real deployment: a facility manager who cannot interrogate a prediction will not act on it.

This project reproduces that baseline on the same public dataset, replaces k-NN with interpretable ensemble models, attaches SHAP explanations, and adapts the approach to a hot-humid climate.

## Objectives

1. Reproduce the published k-NN baseline on CU-BEMS as a reference point.
2. Compare it against Ridge, Random Forest, and gradient boosting under an identical, leakage-free evaluation protocol.
3. Produce global and local SHAP explanations of the best model.
4. Adapt to hot-humid conditions via seasonal clustering, where latent cooling load dominates.

## Dataset

**CU-BEMS** — Chamchuri 5 building, Bangkok. 7 floors, 33 zones, 1-minute resolution, July 2018 – December 2019.

- Kaggle: `claytonmiller/cubems-smart-building-energy-and-iaq-data`
- Citation: Pipattanasomporn, M. et al. (2020). CU-BEMS, smart building electricity consumption and indoor environmental sensor datasets. *Scientific Data* 7:1.

Schema: meters are `z{n}_Light(kW)`, `z{n}_Plug(kW)`, `z{n}_AC{k}(kW)`; sensors are `z{n}_S{k}(degC)`, `(RH%)`, `(lux)`. Floor 1 carries no environmental sensors.

Data is **not** committed to this repository. See `data/README.md` for download instructions.

## Setup

```bash
git clone https://github.com/<user>/smart-building-energy-xai.git
cd smart-building-energy-xai

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place the CSVs in `data/` then run:

```bash
# smoke test, no data required
python src/bems_pipeline.py --synthetic --zone 1 --component light

# real run
python src/bems_pipeline.py --csv data/2019Floor6.csv --zone 1 --component ac --shap
```

## Method notes

- **Chronological split.** A random split leaks future observations into training and inflates R². Train is the leading 80% of the timeline, test the trailing 20%.
- **Hourly aggregation.** The mean of kW readings over an hour equals kWh for that hour.
- **Lag features.** lag-1, lag-24, lag-168 and a 24-hour rolling mean encode thermal inertia, which the published baseline omitted.
- **Metrics.** MAPE is computed on non-zero hours only (night loads approach zero and inflate percentage error), reported alongside sMAPE, RMSE, R², and total-energy relative error.

## Repository layout

```
├── src/                 pipeline code
├── notebooks/           exploration and figure generation
├── docs/                project plan, paper notes
├── results/figures/     committed figures only
├── data/                gitignored; download instructions inside
└── requirements.txt
```

## References

1. Sari, M. et al. (2023). Machine learning-based energy use prediction for the smart building energy management system. *ITcon* 28, 622–645.
2. Gugliermetti, L., Cumo, F., Agostinelli, S. (2024). A future direction of machine learning for building energy management: interpretable models. *Energies* 17(3), 700.
3. Pokkuluri, K. S. et al. (2025). Machine learning-based prediction of energy consumption in smart buildings for sustainable energy management. *JISEM* 10(135).
