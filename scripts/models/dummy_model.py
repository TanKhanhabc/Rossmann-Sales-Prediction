"""Median dummy regression pipeline."""

from sklearn.dummy import DummyRegressor

from scripts.models._pipeline import make_pipeline

PARAM_GRID: dict[str, list[object]] = {}


def build_model():
    return make_pipeline(DummyRegressor(strategy="median"))
