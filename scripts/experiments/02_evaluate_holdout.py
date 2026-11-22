"""Evaluate ALL registered models on the 42-day holdout set.

For each model the best hyper-parameters found in Exp 1 (all_models_params.json)
are used.  Results are written to:
  - reports/all_models_holdout_results.csv   – one row per model
  - reports/all_models_holdout_results.json  – same data with bootstrap CIs
  - reports/final_models.joblib              – fitted artifacts of every model
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from experiment_utils import (
    REPORTS_DIR,
    load_holdout,
    load_all_models,
    predict_sales,
    write_json,
)
from app.model.evaluate_model import score
from app.model.features import make_features
from scripts.models import build_model

BOOTSTRAP_ITERATIONS = 1000


def bootstrap_intervals(actual: np.ndarray, predicted: np.ndarray) -> dict:
    rng = np.random.default_rng(42)
    absolute_error = np.abs(actual - predicted)
    squared_error = np.square(actual - predicted)
    mae_samples = np.empty(BOOTSTRAP_ITERATIONS)
    rmse_samples = np.empty(BOOTSTRAP_ITERATIONS)
    for iteration in range(BOOTSTRAP_ITERATIONS):
        sample = rng.integers(0, len(actual), size=len(actual))
        mae_samples[iteration] = absolute_error[sample].mean()
        rmse_samples[iteration] = np.sqrt(squared_error[sample].mean())
    return {
        "iterations": BOOTSTRAP_ITERATIONS,
        "confidence_level": 0.95,
        "mae_ci95": [float(x) for x in np.percentile(mae_samples, [2.5, 97.5])],
        "rmse_ci95": [float(x) for x in np.percentile(rmse_samples, [2.5, 97.5])],
    }


def run() -> pd.DataFrame:
    print(
        "[EXP2/Giai đoạn 2] Đánh giá 5 mô hình độc lập trên holdout 42 ngày "
        "bằng tham số tốt nhất từ Giai đoạn 1.",
        flush=True,
    )
    _, _, fit, validation, stores, cutoff = load_holdout()
    all_models = load_all_models()   # dict: {model_name: {"params": {...}}}
    actual = validation["Sales"].to_numpy(dtype=float)

    output = []
    fitted_artifacts = {}
    for name, info in all_models.items():
        params = info.get("params", {})
        model = build_model(name, params)
        model.fit(
            make_features(fit, stores),
            np.log1p(fit["Sales"].to_numpy(dtype=float)),
        )
        predicted = predict_sales(model, validation, stores)
        row = {
            "Model": name,
            **score(actual, predicted),
            "bootstrap": bootstrap_intervals(actual, predicted),
        }
        output.append(row)
        fitted_artifacts[name] = model
        print(f"[HOLDOUT] {name}: MAE={row['mae']:.2f}", flush=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORTS_DIR / "all_models_holdout_results.json", {"results": output})
    pd.DataFrame(
        [
            {
                "Model": row["Model"],
                **{k: v for k, v in row.items() if k not in {"Model", "bootstrap"}},
                "MAE_CI95_Lower": row["bootstrap"]["mae_ci95"][0],
                "MAE_CI95_Upper": row["bootstrap"]["mae_ci95"][1],
                "RMSE_CI95_Lower": row["bootstrap"]["rmse_ci95"][0],
                "RMSE_CI95_Upper": row["bootstrap"]["rmse_ci95"][1],
            }
            for row in output
        ]
    ).sort_values("mae").to_csv(REPORTS_DIR / "all_models_holdout_results.csv", index=False)

    joblib.dump(
        {
            "models": fitted_artifacts,
            "trained_through": (cutoff - pd.offsets.Day(1)).strftime("%Y-%m-%d"),
        },
        REPORTS_DIR / "final_models.joblib",
        compress=3,
    )
    write_json(
        REPORTS_DIR / "final_test_metadata.json",
        {
            "fit_rows": int(len(fit)),
            "validation_rows": int(len(validation)),
            "validation_dates": [
                validation["Date"].min().date().isoformat(),
                validation["Date"].max().date().isoformat(),
            ],
            "note": "The 42-day holdout is excluded from fitting; all models use best params from Exp 1.",
        },
    )
    return pd.DataFrame(output)


if __name__ == "__main__":
    run()
