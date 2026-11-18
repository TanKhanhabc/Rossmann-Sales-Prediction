"""Shared data, temporal split, fitting, and scoring utilities for MLOps scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.model.evaluate_model import score
from app.model.features import make_features
from app.services.data_service import get_stores, get_train

REPORTS_DIR = ROOT / "reports"
VALIDATION_DAYS = 42
RANDOM_STATE = 42
CV_SPLITS = 3


def load_holdout():
    """Return all rows, fit rows, open/positive fit rows, validation rows, and stores."""
    raw = get_train().copy()
    stores = get_stores()
    raw["Date"] = pd.to_datetime(raw["Date"], errors="raise")
    raw = raw.sort_values(["Date", "Store"], kind="stable").reset_index(drop=True)
    cutoff = raw["Date"].max() - pd.Timedelta(days=VALIDATION_DAYS - 1)
    fit_rows = raw.loc[raw["Date"].lt(cutoff)].copy()
    validation = raw.loc[raw["Date"].ge(cutoff)].copy()
    fit = fit_rows.loc[fit_rows["Open"].eq(1) & fit_rows["Sales"].gt(0)].copy()
    if fit.empty or validation.empty:
        raise ValueError("The chronological training/validation split is empty.")
    if validation["Date"].nunique() != VALIDATION_DAYS:
        raise ValueError(
            f"Expected {VALIDATION_DAYS} validation dates; found "
            f"{validation['Date'].nunique()}."
        )
    return raw, fit_rows, fit, validation, stores, cutoff


def make_xy(frame: pd.DataFrame, stores: pd.DataFrame):
    features = make_features(frame, stores)
    target = np.log1p(frame["Sales"].to_numpy(dtype=float))
    return features, target


def predict_sales(model, frame: pd.DataFrame, stores: pd.DataFrame) -> np.ndarray:
    features = make_features(frame, stores)
    prediction = np.expm1(model.predict(features))
    prediction = np.maximum(np.asarray(prediction, dtype=float), 0.0)
    closed = frame["Open"].eq(0).to_numpy()
    prediction[closed] = 0.0
    return prediction


def temporal_splits(dates: pd.Series, n_splits: int = CV_SPLITS):
    """Split by unique dates so one calendar day is never divided across folds."""
    unique_dates = np.sort(pd.to_datetime(dates).unique())
    if len(unique_dates) <= n_splits:
        raise ValueError(f"Need more than {n_splits} unique dates for temporal CV.")
    splitter = TimeSeriesSplit(n_splits=n_splits)
    date_positions = np.arange(len(unique_dates))
    splits = []
    date_values = pd.to_datetime(dates).to_numpy()
    for train_dates, valid_dates in splitter.split(date_positions):
        train_set = unique_dates[train_dates]
        valid_set = unique_dates[valid_dates]
        train_idx = np.flatnonzero(np.isin(date_values, train_set))
        valid_idx = np.flatnonzero(np.isin(date_values, valid_set))
        if not len(train_idx) or not len(valid_idx):
            raise ValueError("A temporal CV fold has no training or validation rows.")
        splits.append((train_idx, valid_idx))
    return splits


def original_scale_scorer(metric: str) -> Callable:
    """Build a GridSearchCV scorer that evaluates inverse-log predictions."""
    if metric not in {"mae", "rmse", "r2"}:
        raise ValueError(f"Unsupported metric: {metric}")

    def scorer(estimator, features, target_log):
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        actual = np.expm1(np.asarray(target_log, dtype=float))
        predicted = np.maximum(np.expm1(estimator.predict(features)), 0.0)
        if metric == "mae":
            return -float(mean_absolute_error(actual, predicted))
        if metric == "rmse":
            return -float(np.sqrt(mean_squared_error(actual, predicted)))
        return float(r2_score(actual, predicted))

    return scorer


SCORING = {
    "neg_mae": original_scale_scorer("mae"),
    "neg_rmse": original_scale_scorer("rmse"),
    "r2": original_scale_scorer("r2"),
}


def load_selected_model() -> dict:
    path = REPORTS_DIR / "selected_model.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run `python scripts/experiments/01_compare_models_cv.py` first."
        )
    selected = json.loads(path.read_text(encoding="utf-8"))
    if "model" not in selected or "params" not in selected:
        raise ValueError(f"Invalid model selection file: {path}")
    return selected


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
