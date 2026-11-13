"""Evaluation metrics for model predictions."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def score(y: np.ndarray, pred: np.ndarray) -> dict:
    actual = np.asarray(y, dtype=float)
    guess = np.asarray(pred, dtype=float)
    nonzero = actual > 0
    return {
        "mae": round(float(mean_absolute_error(actual, guess)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, guess))), 2),
        "rmspe": round(float(np.sqrt(np.mean(((actual[nonzero] - guess[nonzero]) / actual[nonzero]) ** 2))), 5) if nonzero.any() else None,
        "wape": round(float(np.abs(actual - guess).sum() / actual.sum()), 5) if actual.sum() else None,
    }
