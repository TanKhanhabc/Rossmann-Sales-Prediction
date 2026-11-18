"""Random forest regression pipeline."""

from sklearn.ensemble import RandomForestRegressor

from scripts.models._pipeline import make_pipeline

PARAM_GRID: list[dict[str, list[object]]] = [
    {
        "model__n_estimators": [32],
        "model__max_depth": [12],
        "model__max_samples": [0.5],
        "model__max_features": [0.7],
    },
]


def build_model():
    return make_pipeline(
        RandomForestRegressor(
            n_estimators=32,
            max_depth=12,
            max_samples=0.5,
            max_features=0.7,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        )
    )
