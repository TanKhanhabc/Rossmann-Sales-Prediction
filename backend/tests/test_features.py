import pandas as pd
from app.model.features import FEATURES, make_features
from app.services.data_service import get_stores, get_train, get_weather


def test_feature_columns_and_no_target_leakage():
    out = make_features(get_train().head(5), get_stores())
    assert len(out) == 5 and list(out.columns) == FEATURES
    assert "Sales" not in out and "Customers" not in out
    assert not out.isna().any().any()


def test_promo2_month_flag_and_closed_date():
    schedule = pd.DataFrame([{"Store": 2, "Date": "2015-07-01", "Promo": 1, "SchoolHoliday": 0, "StateHoliday": "0"},
                             {"Store": 2, "Date": "2015-08-01", "Promo": 0, "SchoolHoliday": 1, "StateHoliday": "0"}])
    out = make_features(schedule, get_stores(), get_weather())
    assert int(out.iloc[0].Promo2MonthActive) == 1
    assert int(out.iloc[1].Promo2MonthActive) == 0
    assert int(out.iloc[0].DayOfWeek) == 3


def test_weather_features_fall_back_to_state_history_for_unmatched_dates():
    schedule = pd.DataFrame([{
        "Store": 2, "Date": "2020-01-01", "Promo": 0,
        "SchoolHoliday": 0, "StateHoliday": "0",
    }])

    out = make_features(schedule, get_stores(), get_weather())

    assert list(out.columns) == FEATURES
    assert not out.isna().any().any()
    assert int(out.iloc[0].WeatherMissingCount) == 21
