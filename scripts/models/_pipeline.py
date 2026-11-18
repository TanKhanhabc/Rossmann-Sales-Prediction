"""Shared preprocessing for experiment estimators."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from app.model.features import CATEGORICAL, FEATURES

NUMERIC = [column for column in FEATURES if column not in CATEGORICAL]


def make_preprocessor(*, ordinal: bool = False) -> ColumnTransformer:
    if ordinal:
        categorical = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "ordinal",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        dtype=np.float32,
                    ),
                ),
            ]
        )
    else:
        categorical = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", dtype=np.float32)),
            ]
        )

    numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical, CATEGORICAL),
            ("numeric", numeric, NUMERIC),
        ],
        remainder="drop",
    )


def make_pipeline(
    regressor,
    *,
    scale: bool = False,
    ordinal: bool | None = None,
) -> Pipeline:
    # Tự động dùng OrdinalEncoder nếu không scale (các mô hình cây),
    # và dùng OneHotEncoder nếu có scale (mô hình tuyến tính như Ridge)
    use_ordinal = (not scale) if ordinal is None else ordinal

    steps = [("preprocess", make_preprocessor(ordinal=use_ordinal))]
    if scale:
        steps.append(("scale", StandardScaler(with_mean=False)))
    steps.append(("model", regressor))
    return Pipeline(steps=steps)