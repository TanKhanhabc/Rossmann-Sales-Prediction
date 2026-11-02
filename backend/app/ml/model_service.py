"""Training, evaluation, persistence, inference."""
from __future__ import annotations
import json
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from app.ml.features import FEATURES, CATEGORICAL, make_features
from app.services.data_service import MODEL_FILE, METRICS_FILE, get_stores, get_train


def build_model(n_estimators: int = 420) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=n_estimators, learning_rate=0.055, num_leaves=38,
        max_depth=-1, min_child_samples=45, reg_lambda=2.0,
        colsample_bytree=0.90, subsample=0.90, subsample_freq=1,
        random_state=42, n_jobs=4, verbosity=-1,
    )


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


def predict_frame(model: LGBMRegressor, df: pd.DataFrame, store_df: pd.DataFrame) -> np.ndarray:
    feats = make_features(df, store_df)
    guess = np.expm1(model.predict(feats))
    open_flag = df["Open"].fillna(1).astype(int).eq(1).to_numpy() if "Open" in df else np.ones(len(df), dtype=bool)
    return np.where(open_flag, np.maximum(0, guess), 0.0)


def train_and_evaluate(validation_days: int = 42, n_estimators: int = 420) -> dict:
    raw = get_train()
    stores = get_stores()
    cutoff = raw.Date.max() - pd.Timedelta(days=validation_days - 1)
    fit_mask = raw.Date.lt(cutoff)
    valid_mask = ~fit_mask
    positive_fit = fit_mask & raw.Open.eq(1) & raw.Sales.gt(0)
    fit_df = raw.loc[positive_fit].copy()
    val_df = raw.loc[valid_mask].copy()
    print(f"[TRAIN] Split: fit {len(fit_df):,} open/sales>0, validation {len(val_df):,} records", flush=True)
    features_train = make_features(fit_df, stores)
    model = build_model(n_estimators)
    model.fit(features_train, np.log1p(fit_df.Sales), categorical_feature=CATEGORICAL)
    prediction = predict_frame(model, val_df, stores)
    # Leakage-safe baseline: historical Store x weekday x promo mean from PRE-validation only.
    history = raw.loc[positive_fit]
    baseline_stats = history.groupby(["Store", "DayOfWeek", "Promo"])["Sales"].mean()
    store_mean = history.groupby("Store")["Sales"].mean()
    baseline = []
    global_mean = float(history.Sales.mean())
    for row in val_df[["Store", "DayOfWeek", "Promo", "Open"]].itertuples(index=False):
        fallback = store_mean.get(row.Store, global_mean)
        baseline.append(0.0 if row.Open == 0 else float(baseline_stats.get((row.Store, row.DayOfWeek, row.Promo), fallback)))
    metrics = {
        "model": "LightGBM regression on log1p(Sales)", "target": "Sales per store per day",
        "train_dates": [raw.loc[fit_mask, "Date"].min().date().isoformat(), (cutoff - pd.Timedelta(days=1)).date().isoformat()],
        "validation_dates": [cutoff.date().isoformat(), raw.Date.max().date().isoformat()],
        "train_rows": int(positive_fit.sum()), "validation_rows": len(val_df),
        "validation": score(val_df.Sales.to_numpy(), prediction),
        "baseline": score(val_df.Sales.to_numpy(), np.asarray(baseline)),
        "features": FEATURES,
        "important_features": sorted([{"feature": k, "importance": int(v)} for k, v in zip(FEATURES, model.feature_importances_)], key=lambda o: -o["importance"])[:12],
        "notes": ["42-day chronological holdout; no random split.",
                  "Sales and Customers are explicitly excluded from input features.",
                  "Closed stores yield 0 by business rule; missing test Open is estimated.",
                  "The validation model was trained only on dates before the validation interval."],
    }
    print("[TRAIN] Validation:", json.dumps(metrics["validation"]),flush=True)
    print("[TRAIN] Baseline:", json.dumps(metrics["baseline"]),flush=True)
    # Train production model on ALL historical positive open rows.
    full_fit = raw.loc[raw.Open.eq(1) & raw.Sales.gt(0)]
    print(f"[TRAIN] Full model: {len(full_fit):,} rows...",flush=True)
    production = build_model(n_estimators)
    production.fit(make_features(full_fit, stores), np.log1p(full_fit.Sales), categorical_feature=CATEGORICAL)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": production, "features": FEATURES, "metrics": metrics,
                 "trained_through": raw.Date.max().date().isoformat()}, MODEL_FILE, compress=3)
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    load_model.cache_clear()
    print("[TRAIN] Saved model to", MODEL_FILE, flush=True)
    return metrics


@lru_cache(maxsize=1)
def load_model() -> dict:
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Chưa có mô hình. Hãy chạy: python scripts/analysis/run_evaluation.py")
    return joblib.load(MODEL_FILE)


def infer(df: pd.DataFrame) -> np.ndarray:
    return predict_frame(load_model()["model"], df, get_stores())
