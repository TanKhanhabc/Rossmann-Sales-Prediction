import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from app.model.features import FEATURES
from app.model.predict_model import load_model, predict_frame
from app.services.data_service import get_stores, get_test


def test_saved_model_matches_feature_schema():
    artifact = load_model()

    assert isinstance(artifact["model"], LGBMRegressor)
    assert artifact["features"] == FEATURES


def test_model_predictions_are_valid_and_zero_when_store_is_closed():
    test_data = get_test()
    schedule = pd.concat(
        [
            test_data.loc[test_data.Open.eq(1)].head(1),
            test_data.loc[test_data.Open.eq(0)].head(1),
        ],
        ignore_index=True,
    )

    predictions = predict_frame(load_model()["model"], schedule, get_stores())

    assert predictions.shape == (2,)
    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()
    assert predictions[1] == 0
