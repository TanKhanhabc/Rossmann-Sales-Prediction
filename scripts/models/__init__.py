"""Reusable estimators and compact parameter grids for experiments."""

from __future__ import annotations

from sklearn.pipeline import Pipeline

from scripts.models.decision_tree_model import build_model as build_decision_tree
from scripts.models.decision_tree_model import PARAM_GRID as DECISION_TREE_GRID
from scripts.models.dummy_model import build_model as build_dummy
from scripts.models.dummy_model import PARAM_GRID as DUMMY_GRID
from scripts.models.lgbm_model import build_model as build_lgbm
from scripts.models.lgbm_model import PARAM_GRID as LGBM_GRID
from scripts.models.random_forest_model import build_model as build_random_forest
from scripts.models.random_forest_model import PARAM_GRID as RANDOM_FOREST_GRID
from scripts.models.ridge_model import build_model as build_ridge
from scripts.models.ridge_model import PARAM_GRID as RIDGE_GRID

ModelParameterGrid = dict[str, list[object]] | list[dict[str, list[object]]]
ModelSpec = tuple[Pipeline, ModelParameterGrid]


def get_models() -> dict[str, ModelSpec]:
    """Create fresh pipelines and grids, keyed by stable report identifiers."""
    return {
        "dummy": (build_dummy(), DUMMY_GRID),
        "ridge": (build_ridge(), RIDGE_GRID),
        "decision_tree": (build_decision_tree(), DECISION_TREE_GRID),
        "random_forest": (build_random_forest(), RANDOM_FOREST_GRID),
        "lightgbm": (build_lgbm(), LGBM_GRID),
    }


def build_model(name: str, params: dict[str, object] | None = None) -> Pipeline:
    """Build one named pipeline and optionally apply GridSearchCV parameters."""
    try:
        model = get_models()[name][0]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {name}") from exc
    if params:
        model.set_params(**params)
    return model


__all__ = ["ModelParameterGrid", "ModelSpec", "build_model", "get_models"]
