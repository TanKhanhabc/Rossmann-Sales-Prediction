"""LightGBM regression pipeline matching the production model's baseline settings."""

from lightgbm import LGBMRegressor

from scripts.models._pipeline import make_pipeline

PARAM_GRID: list[dict[str, list[object]]] = [
    {
        "model__n_estimators": [420],
        "model__num_leaves": [38],
        "model__learning_rate": [0.055],
        "model__max_depth": [-1],
        "model__min_child_samples": [45],
        "model__reg_lambda": [2.0],
        "model__colsample_bytree": [0.90],
        "model__subsample": [0.90],
        "model__subsample_freq": [1],
        "model__random_state": [42],
    },
]


def build_model():
    return make_pipeline(
        LGBMRegressor(
            n_estimators=420,
            num_leaves=38,
            learning_rate=0.055,
            max_depth=-1,
            min_child_samples=45,
            reg_lambda=2.0,
            colsample_bytree=0.90,
            subsample=0.90,
            subsample_freq=1,
            random_state=42,
            n_jobs=1,
            verbosity=-1,
        )
    )
