"""Tune each registered pipeline with chronological cross-validation."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV
from threadpoolctl import threadpool_limits

from experiment_utils import (
    REPORTS_DIR,
    SCORING,
    make_xy,
    load_holdout,
    temporal_splits,
    write_json,
)
from scripts.models import get_models

CV_N_JOBS = 1
MAX_CONFIGURATIONS = 1
RANDOM_STATE = 42

COMPLEXITY = {
    "dummy": 0,
    "ridge": 1,
    "decision_tree": 2,
    "random_forest": 3,
    "lightgbm": 4,
}


def run() -> pd.DataFrame:
    _, _, fit, _, stores, cutoff = load_holdout()
    features, target = make_xy(fit, stores)
    cv = temporal_splits(fit["Date"])
    models = get_models()
    results = []
    run_started = perf_counter()
    print(
        f"[CV] Fit rows: {len(fit):,}; dates: {fit['Date'].min().date()} to "
        f"{fit['Date'].max().date()}; 42-day holdout from {cutoff.date()} excluded. "
        f"Folds: {len(cv)}; parallel workers: {CV_N_JOBS}. "
        f"Randomized-search cap: {MAX_CONFIGURATIONS} configuration/model; "
        "fewer configurations can change the selected parameters and results.",
        flush=True,
    )
    for model_number, (name, (pipeline, parameter_grid)) in enumerate(models.items(), start=1):
        candidate_count = len(ParameterGrid(parameter_grid))
        configurations_to_try = min(MAX_CONFIGURATIONS, candidate_count)
        model_started = perf_counter()
        print(
            f"[CV] Model {model_number}/{len(models)}: {name}; "
            f"{configurations_to_try}/{candidate_count} configurations x {len(cv)} folds "
            f"(plus one refit) starting.",
            flush=True,
        )
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=parameter_grid,
            n_iter=configurations_to_try,
            scoring=SCORING,
            refit="neg_mae",
            cv=cv,
            n_jobs=CV_N_JOBS,
            pre_dispatch=CV_N_JOBS,
            random_state=RANDOM_STATE,
            return_train_score=True,
            error_score="raise",
            verbose=2,
        )
        with threadpool_limits(limits=CV_N_JOBS):
            search.fit(features, target)
        cv_result = search.cv_results_
        index = search.best_index_
        fold_mae = np.asarray(
            [-cv_result[f"split{fold}_test_neg_mae"][index] for fold in range(len(cv))]
        )
        results.append(
            {
                "Model": name,
                "Train_MAE": -float(cv_result["mean_train_neg_mae"][index]),
                "Val_MAE": float(fold_mae.mean()),
                "Val_MAE_SD": float(fold_mae.std(ddof=1)) if len(fold_mae) > 1 else 0.0,
                "RMSE": -float(cv_result["mean_test_neg_rmse"][index]),
                "R2": float(cv_result["mean_test_r2"][index]),
                "Best_Params": json.dumps(search.best_params_, sort_keys=True),
            }
        )
        print(
            f"[CV] {name} complete in {perf_counter() - model_started:.1f}s: "
            f"Val MAE={results[-1]['Val_MAE']:.2f} "
            f"+/- {results[-1]['Val_MAE_SD']:.2f}; best params={search.best_params_}",
            flush=True,
        )

    table = pd.DataFrame(results).sort_values("Val_MAE").reset_index(drop=True)
    best = table.iloc[0]
    threshold = float(best["Val_MAE"] + best["Val_MAE_SD"])
    eligible = table.loc[table["Val_MAE"].le(threshold)].copy()
    selected = eligible.sort_values(
        ["Model"], key=lambda column: column.map(COMPLEXITY)
    ).iloc[0]
    table["Within_1SD"] = table["Val_MAE"].le(threshold)
    table["Selected"] = table["Model"].eq(selected["Model"])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS_DIR / "cv_results.csv", index=False)
    write_json(
        REPORTS_DIR / "selected_model.json",
        {
            "model": selected["Model"],
            "params": json.loads(selected["Best_Params"]),
            "selection_rule": "simplest model within one SD of the lowest mean validation MAE",
            "best_mean_val_mae": float(best["Val_MAE"]),
            "one_sd_threshold": threshold,
            "validation_start": cutoff.date().isoformat(),
            "cv_splits": len(cv),
            "cv_validation_metric": "MAE on original Sales scale",
        },
    )
    plot_results(table, threshold, REPORTS_DIR / "figures" / "cv_one_sd.png")
    print(
        f"[CV] Complete in {perf_counter() - run_started:.1f}s. "
        "Wrote reports/cv_results.csv and reports/selected_model.json",
        flush=True,
    )
    return table


def plot_results(table: pd.DataFrame, threshold: float, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = table.sort_values("Val_MAE", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#E45756" if selected else "#4C78A8" for selected in ordered["Selected"]]
    ax.bar(
        ordered["Model"],
        ordered["Val_MAE"],
        yerr=ordered["Val_MAE_SD"],
        capsize=4,
        color=colors,
    )
    ax.axhline(threshold, color="#F58518", linestyle="--", label="Best MAE + 1 SD")
    ax.set_ylabel("Validation MAE (Sales)")
    ax.set_title("Chronological CV: validation MAE and one-SD selection threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
