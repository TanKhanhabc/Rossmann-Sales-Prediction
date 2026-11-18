"""Measure holdout permutation importance in original-Sales MAE units."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from experiment_utils import REPORTS_DIR, load_holdout, load_selected_model, make_xy
from scripts.models import build_model


def run() -> pd.DataFrame:
    _, _, fit, validation, stores, _ = load_holdout()
    selected = load_selected_model()
    model = build_model(selected["model"], selected["params"])
    train_features, train_target = make_xy(fit, stores)
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
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    figure_dir = REPORTS_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)
    plot_importance(importance, figure_dir / "permutation_importance.png")
    print(f"[IMPORTANCE] Saved {len(importance)} features to reports/permutation_importance.csv")
    return importance


def plot_importance(importance: pd.DataFrame, output) -> None:
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
    ax.set_title("Permutation feature importance (mean +/- 1 SD, 10 repeats)")
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
