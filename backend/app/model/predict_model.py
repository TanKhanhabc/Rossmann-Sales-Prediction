"""Inference helpers shared by API and scripts."""
from __future__ import annotations

from functools import lru_cache
from threading import Lock
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd

from app.model.features import WEATHER_FEATURES, make_features
from app.services.data_service import MODEL_FILE, get_stores, get_weather

if TYPE_CHECKING:
    from lightgbm import LGBMRegressor

_INFERENCE_LOCK = Lock()


def predict_frame(model: LGBMRegressor, df: pd.DataFrame, store_df: pd.DataFrame) -> np.ndarray:
    weather = None if set(WEATHER_FEATURES).issubset(df.columns) else get_weather()
    feats = make_features(df, store_df, weather)
    guess = np.expm1(model.predict(feats))
    open_flag = df["Open"].fillna(1).astype(int).eq(1).to_numpy() if "Open" in df else np.ones(len(df), dtype=bool)
    return np.where(open_flag, np.maximum(0, guess), 0.0)


@lru_cache(maxsize=1)
def load_model() -> dict:
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Chưa có mô hình. Hãy chạy: python scripts/evaluation/run_evaluation.py")
    return joblib.load(MODEL_FILE)


def infer(df: pd.DataFrame) -> np.ndarray:
    with _INFERENCE_LOCK:
        return predict_frame(load_model()["model"], df, get_stores())
