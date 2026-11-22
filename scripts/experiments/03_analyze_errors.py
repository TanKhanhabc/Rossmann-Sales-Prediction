"""Report holdout subgroup errors and residual-based prediction intervals for ALL models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiment_utils import (
    REPORTS_DIR,
    load_holdout,
    load_all_models,
    predict_sales,
    temporal_splits,
)
from app.model.evaluate_model import score
from app.model.features import make_features
from scripts.models import build_model


def run() -> dict[str, dict[str, pd.DataFrame]]:
    """Run error analysis for every model in all_models_params.json.

    Returns a nested dict: {model_name: {subgroup_column: DataFrame}}.
    """
    print(
        "[EXP3/Giai đoạn 3] Phân tích sai số theo nhóm và khoảng dự đoán "
        "cho từng mô hình.",
        flush=True,
    )
    _, _, fit, validation, stores, _ = load_holdout()
    all_models = load_all_models()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, pd.DataFrame]] = {}
    for model_name, info in all_models.items():
        print(f"[EXP3] Analysing errors for model: {model_name}", flush=True)
        params = info.get("params", {})
        subgroup_tables = _run_one_model(model_name, params, fit, validation, stores)
        all_results[model_name] = subgroup_tables

    return all_results


def _run_one_model(
    model_name: str,
    params: dict,
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    stores: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Error analysis pipeline for a single model.

    Outputs are saved to reports/<model_name>/ so each model has its own folder.
    """
    model_dir = REPORTS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # --- OOF residuals để tính prediction interval ---
    oof_prediction = np.full(len(fit), np.nan, dtype=float)
    for train_idx, valid_idx in temporal_splits(fit["Date"]):
        model = build_model(model_name, params)
        train_fold = fit.iloc[train_idx]
        valid_fold = fit.iloc[valid_idx]
        model.fit(
            make_features(train_fold, stores),
            np.log1p(train_fold["Sales"].to_numpy(dtype=float)),
        )
        oof_prediction[valid_idx] = predict_sales(model, valid_fold, stores)

    valid_oof = np.isfinite(oof_prediction)
    residuals = fit["Sales"].to_numpy(dtype=float)[valid_oof] - oof_prediction[valid_oof]
    if not len(residuals):
        raise ValueError(f"[{model_name}] No out-of-fold residuals were produced.")
    lower_residual, upper_residual = np.percentile(residuals, [10, 90])

    # --- Train trên toàn bộ fit → dự báo validation ---
    model = build_model(model_name, params)
    model.fit(
        make_features(fit, stores),
        np.log1p(fit["Sales"].to_numpy(dtype=float)),
    )
    prediction = predict_sales(model, validation, stores)
    lower = np.maximum(prediction + lower_residual, 0.0)
    upper = np.maximum(prediction + upper_residual, 0.0)
    closed = validation["Open"].eq(0).to_numpy()
    lower[closed] = 0.0
    upper[closed] = 0.0

    evaluated = validation.reset_index(drop=True).copy()
    evaluated["Prediction"] = prediction
    evaluated["Absolute_Error"] = np.abs(
        evaluated["Sales"].to_numpy(dtype=float) - prediction
    )
    validation_features = make_features(validation, stores)
    evaluated["StoreType"] = validation_features["StoreType"].astype(str).to_numpy()
    evaluated["DayOfWeek"] = validation_features["DayOfWeek"].to_numpy()

    # --- Subgroup MAE ---
    subgroup_tables = {}
    for column in ("StoreType", "DayOfWeek", "Promo"):
        table = grouped_mae(evaluated, column)
        subgroup_tables[column] = table
        table.to_csv(model_dir / f"subgroup_mae_{column.lower()}.csv", index=False)

    # --- Top 10 lỗi lớn nhất ---
    error_columns = [
        column for column in ("Date", "Store", "StoreType", "DayOfWeek", "Promo", "Open", "Sales")
        if column in evaluated.columns
    ]
    top_errors = evaluated.nlargest(10, "Absolute_Error")[error_columns + [
        "Prediction", "Absolute_Error"
    ]]
    top_errors.to_csv(model_dir / "top_10_errors.csv", index=False)

    # --- Prediction interval ---
    interval_frame = evaluated[["Date", "Store", "Sales", "Prediction"]].copy()
    interval_frame["P10"] = lower
    interval_frame["P90"] = upper
    interval_frame["Inside_P10_P90"] = interval_frame["Sales"].between(
        interval_frame["P10"], interval_frame["P90"]
    )
    interval_frame.to_csv(model_dir / "validation_prediction_intervals.csv", index=False)
    pd.DataFrame(
        [{
            "model": model_name,
            "residual_oof_rows": int(len(residuals)),
            "residual_p10": float(lower_residual),
            "residual_p90": float(upper_residual),
            "validation_empirical_coverage": float(interval_frame["Inside_P10_P90"].mean()),
            "interval_method": "additive out-of-fold residual quantiles (P10, P90), clipped at zero",
        }]
    ).to_csv(model_dir / "prediction_interval_summary.csv", index=False)

    print(
        f"[EXP3] {model_name}: coverage={interval_frame['Inside_P10_P90'].mean():.2%}, "
        f"reports saved to reports/{model_name}/",
        flush=True,
    )
    return subgroup_tables


def grouped_mae(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for group, subset in frame.groupby(column, observed=True):
        rows.append(
            {
                column: group,
                "Rows": len(subset),
                "MAE": score(
                    subset["Sales"].to_numpy(dtype=float),
                    subset["Prediction"].to_numpy(dtype=float),
                )["mae"],
            }
        )
    return pd.DataFrame(rows).sort_values("MAE", ascending=False)


if __name__ == "__main__":
    run()

