"""Measure holdout permutation importance in original-Sales MAE units for ALL models."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from experiment_utils import REPORTS_DIR, load_holdout, load_all_models, make_xy
from scripts.models import build_model


def run() -> dict[str, pd.DataFrame]:
    """Run permutation importance for every model in all_models_params.json.

    Returns a dict: {model_name: importance_DataFrame}.
    """
    print(
        "[EXP4/Giai đoạn 4] Đánh giá permutation importance trên holdout "
        "để giải thích đóng góp của từng đặc trưng.",
        flush=True,
    )
    _, _, fit, validation, stores, _ = load_holdout()
    all_models = load_all_models()

    all_importance: dict[str, pd.DataFrame] = {}
    for model_name, info in all_models.items():
        print(f"[EXP4] Computing permutation importance for: {model_name}", flush=True)
        params = info.get("params", {})
        importance = _run_one_model(model_name, params, fit, validation, stores)
        all_importance[model_name] = importance

    return all_importance


def _run_one_model(
    model_name: str,
    params: dict,
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    stores: pd.DataFrame,
) -> pd.DataFrame:
    """Permutation importance pipeline for a single model.

    Outputs are saved to reports/<model_name>/ so each model has its own folder.
    """
    model_dir = REPORTS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = model_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    train_features, train_target = make_xy(fit, stores)
    model = build_model(model_name, params)
    model.fit(train_features, train_target)

    open_validation = validation.loc[validation["Open"].eq(1)].copy()
    if len(open_validation) > 20000:
        open_validation = open_validation.sample(n=20000, random_state=42)
    features, target = make_xy(open_validation, stores)

    result = permutation_importance(
        model,
        features,
        target,
        scoring=lambda estimator, X, y: -float(
            np.mean(
                np.abs(
                    np.expm1(np.asarray(y, dtype=float))
                    - np.maximum(np.expm1(estimator.predict(X)), 0.0)
                )
            )
        ),
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "Feature": features.columns,
            "Importance_MAE_Increase": result.importances_mean,
            "Importance_SD": result.importances_std,
        }
    ).sort_values("Importance_MAE_Increase", ascending=False)

    importance.to_csv(model_dir / "permutation_importance.csv", index=False)
    plot_importance(importance, figure_dir / "permutation_importance.png", model_name)
    print(
        f"[EXP4] {model_name}: {len(importance)} features saved to reports/{model_name}/permutation_importance.csv",
        flush=True,
    )
    return importance


def plot_importance(importance: pd.DataFrame, output, model_name: str = "") -> None:
    shown = importance.head(25).sort_values("Importance_MAE_Increase")
    fig, ax = plt.subplots(figsize=(11, max(6, len(shown) * 0.28)))
    ax.barh(
        shown["Feature"],
        shown["Importance_MAE_Increase"],
        xerr=shown["Importance_SD"],
        capsize=2,
        color="#4C78A8",
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Increase in validation MAE when permuted (Sales)")
    title = f"Permutation feature importance — {model_name} (mean +/- 1 SD, 10 repeats)"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()

