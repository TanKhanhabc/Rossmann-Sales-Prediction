"""Decision tree regression pipeline."""

from sklearn.tree import DecisionTreeRegressor

from scripts.models._pipeline import make_pipeline

PARAM_GRID: list[dict[str, list[object]]] = [
    {"model__max_depth": [12], "model__min_samples_leaf": [20]},
    {"model__max_depth": [None], "model__min_samples_leaf": [50]},
]


def build_model():
    return make_pipeline(DecisionTreeRegressor(random_state=42))
