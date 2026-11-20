import pandas as pd
from app.model.features import (
    FEATURES,
    WEATHER_EVENTS,
    WEATHER_FEATURES,
    WEATHER_NUMERIC_COLUMNS,
    make_features,
)
from app.services.data_service import get_stores, get_train, get_weather


def _weather_observation(state, date, temperature):
    row = {
        "State": state,
        "Date": date,
        **{column: 10.0 for column in WEATHER_NUMERIC_COLUMNS},
        "WeatherMissingCount": 0,
        "WindDirSin": 0.0,
        "WindDirCos": 1.0,
        **{f"Weather{event}": 0 for event in WEATHER_EVENTS},
    }
    row["Mean_TemperatureC"] = temperature
    return row


def _weather_scenario_features(weather, target_date="2024-03-08"):
    stores = get_stores()
    store = stores.iloc[0]
    schedule = pd.DataFrame([{
        "Store": int(store.Store),
        "Date": target_date,
        "Promo": 0,
        "SchoolHoliday": 0,
        "StateHoliday": "0",
    }])
    return make_features(schedule, stores, pd.DataFrame(weather))


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


def test_missing_weather_uses_same_state_calendar_day_from_past_years():
    state = get_stores().iloc[0].State
    weather = [
        _weather_observation(state, "2018-03-08", 10.0),
        _weather_observation(state, "2019-03-01", 20.0),
        _weather_observation("another-state", "2019-03-08", 30.0),
        _weather_observation(state, "2025-03-08", 1000.0),
    ]

    out = _weather_scenario_features(weather)

    assert out.iloc[0].Mean_TemperatureC == 10.0
    assert list(out.columns) == FEATURES


def test_missing_weather_falls_back_to_same_state_month_then_global_past():
    state = get_stores().iloc[0].State
    month_weather = [
        _weather_observation(state, "2020-03-01", 20.0),
        _weather_observation("another-state", "2020-02-01", 30.0),
        _weather_observation(state, "2025-03-08", 1000.0),
    ]
    global_weather = [
        _weather_observation(state, "2020-02-01", 10.0),
        _weather_observation("another-state", "2020-02-02", 30.0),
        _weather_observation(state, "2025-03-08", 1000.0),
    ]

    month_out = _weather_scenario_features(month_weather)
    global_out = _weather_scenario_features(global_weather)

    assert month_out.iloc[0].Mean_TemperatureC == 20.0
    assert global_out.iloc[0].Mean_TemperatureC == 20.0
    assert list(month_out.columns) == FEATURES
    assert list(global_out.columns) == FEATURES
    assert set(WEATHER_FEATURES).issubset(global_out.columns)
