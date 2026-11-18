"""Standardized Ridge regression pipeline."""

from sklearn.linear_model import Ridge

from scripts.models._pipeline import make_pipeline

PARAM_GRID: dict[str, list[object]] = {"model__alpha": [1.0, 10.0]}


def build_model():
    return make_pipeline(Ridge(), scale=True)
