"""Regression metrics used for CU-BEMS load forecasting."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    """Return complementary error, fit, and energy-bias measures.

    MAPE and a20 use only hours with non-zero measured load, avoiding the
    undefined/inflated percentage errors caused by zero or near-zero loads.
    Percentage metrics are expressed as percentages (not fractions).
    """
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if actual.size == 0 or actual.size != predicted.size:
        raise ValueError("y_true and y_pred must be non-empty arrays of equal length")

    error = predicted - actual
    mean_actual = float(np.mean(actual))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))

    nonzero = actual != 0
    if np.any(nonzero):
        mape = float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100)
        a20 = float(np.mean(np.abs(error[nonzero] / actual[nonzero]) <= 0.20) * 100)
    else:
        mape = a20 = float("nan")

    denominator = np.abs(actual) + np.abs(predicted)
    smape_terms = np.divide(
        2 * np.abs(error), denominator,
        out=np.zeros_like(error), where=denominator != 0,
    )
    smape = float(np.mean(smape_terms) * 100)

    variance = float(np.var(actual))
    vaf = float((1 - np.var(error) / variance) * 100) if variance else float("nan")
    ioa_denominator = float(np.sum((np.abs(predicted - mean_actual) +
                                    np.abs(actual - mean_actual)) ** 2))
    ioa = (float(1 - np.sum(error ** 2) / ioa_denominator)
           if ioa_denominator else float("nan"))

    total_actual = float(np.sum(actual))
    total_predicted = float(np.sum(predicted))
    if mean_actual:
        cvrmse = rmse / abs(mean_actual) * 100
        nmbe = float(np.sum(error) / (actual.size * mean_actual) * 100)
    else:
        cvrmse = nmbe = float("nan")
    total_bias = ((total_predicted - total_actual) / total_actual * 100
                  if total_actual else float("nan"))

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": float(mean_squared_error(actual, predicted)),
        "rmse": rmse,
        "r2": float(r2_score(actual, predicted)) if actual.size > 1 else float("nan"),
        "vaf_pct": vaf,
        "a20_pct": a20,
        "ioa": ioa,
        "cvrmse_pct": cvrmse,
        "nmbe_pct": nmbe,
        "mape_nonzero_pct": mape,
        "smape_pct": smape,
        "total_energy_bias_pct": total_bias,
        "total_energy_abs_error_pct": abs(total_bias) if np.isfinite(total_bias) else float("nan"),
    }
