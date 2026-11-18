"""Evaluate the selected model and one reference model on the 42-day holdout."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from experiment_utils import (
    REPORTS_DIR,
    load_holdout,
    load_selected_model,
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
    _, _, fit, validation, stores, cutoff = load_holdout()
    selected = load_selected_model()
    control_name = "ridge" if selected["model"] == "dummy" else "dummy"
    models = {
        selected["model"]: build_model(selected["model"], selected["params"]),
        control_name: build_model(control_name),
    }
    actual = validation["Sales"].to_numpy(dtype=float)
    output = []
    fitted_artifacts = {}
    for name, model in models.items():
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
        print(f"[FINAL] {name}: {json.dumps(row, ensure_ascii=False)}", flush=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORTS_DIR / "final_test_results.json", {"results": output})
    pd.DataFrame(
        [
            {
                "Model": row["Model"],
                **{key: value for key, value in row.items() if key not in {"Model", "bootstrap"}},
                "MAE_CI95_Lower": row["bootstrap"]["mae_ci95"][0],
                "MAE_CI95_Upper": row["bootstrap"]["mae_ci95"][1],
                "RMSE_CI95_Lower": row["bootstrap"]["rmse_ci95"][0],
                "RMSE_CI95_Upper": row["bootstrap"]["rmse_ci95"][1],
            }
            for row in output
        ]
    ).to_csv(REPORTS_DIR / "final_test_results.csv", index=False)
    joblib.dump(
        {
            "selected_model": fitted_artifacts[selected["model"]],
            "control_model": fitted_artifacts[control_name],
            "selected_model_name": selected["model"],
            "control_model_name": control_name,
            "trained_through": (cutoff - pd.Timedelta(days=1)).date().isoformat(),
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
            "note": "The 42-day holdout is excluded from fitting; both models are fit on all eligible pre-holdout training rows.",
        },
    )
    return pd.DataFrame(output)


if __name__ == "__main__":
    run()
