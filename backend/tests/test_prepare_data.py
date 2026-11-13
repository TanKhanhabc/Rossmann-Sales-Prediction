import pandas as pd

from scripts.preprocessing.prepare_data import WEATHER_NUMERIC_COLUMNS, prepare_data


def test_prepare_data_cleans_inputs_without_modifying_raw_files(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    merged_dir = tmp_path / "merged"
    external_dir = tmp_path / "external"
    raw_dir.mkdir()
    external_dir.mkdir()
    pd.DataFrame([{
        "Store": 1, "DayOfWeek": 1, "Date": "2015-01-05", "Sales": 10,
        "Customers": 1, "Open": 1, "Promo": None, "StateHoliday": None,
        "SchoolHoliday": None,
    }]).to_csv(raw_dir / "train.csv", index=False)
    pd.DataFrame([
        {"Id": 1, "Store": 1, "DayOfWeek": 1, "Date": "2015-01-05",
         "Open": None, "Promo": None, "StateHoliday": None, "SchoolHoliday": None},
        {"Id": 2, "Store": 1, "DayOfWeek": 7, "Date": "2015-01-11",
         "Open": None, "Promo": 0, "StateHoliday": "0", "SchoolHoliday": 0},
    ]).to_csv(raw_dir / "test.csv", index=False)
    pd.DataFrame([{
        "Store": 1, "StoreType": "a", "Assortment": "a",
        "CompetitionDistance": None, "CompetitionOpenSinceMonth": None,
        "CompetitionOpenSinceYear": None, "Promo2": None,
        "Promo2SinceWeek": None, "Promo2SinceYear": None, "PromoInterval": None,
    }]).to_csv(raw_dir / "store.csv", index=False)
    pd.DataFrame([{"StateName": "Testland", "State": "TS"}]).to_csv(
        external_dir / "state_names.csv", index=False
    )
    pd.DataFrame([{"Store": 1, "State": "TS"}]).to_csv(
        external_dir / "store_states.csv", index=False
    )
    weather_rows = []
    for day, event, gust in (("2015-01-05", "Rain-Fog", None), ("2015-01-11", "", 12)):
        row = {
            "file": "Testland", "Date": day, "Events": event,
            **{column: 1 for column in WEATHER_NUMERIC_COLUMNS},
        }
        row["WindDirDegrees"] = 180
        row["Max_Gust_SpeedKm_h"] = gust
        weather_rows.append(row)
    pd.DataFrame(weather_rows).to_csv(external_dir / "weather.csv", index=False)
    raw_test_before = (raw_dir / "test.csv").read_bytes()

    report = prepare_data(raw_dir, processed_dir, merged_dir, external_dir)

    prepared_test = pd.read_csv(processed_dir / "test.csv", dtype={"StateHoliday": str})
    prepared_train = pd.read_csv(processed_dir / "train.csv")
    prepared_store = pd.read_csv(processed_dir / "store.csv", keep_default_na=False)
    prepared_train_merged = pd.read_csv(merged_dir / "train_merged.csv")
    prepared_test_merged = pd.read_csv(merged_dir / "test_merged.csv")
    prepared_weather = pd.read_csv(processed_dir / "external" / "weather.csv")
    assert prepared_test["Open"].tolist() == [1, 0]
    assert prepared_test["Promo"].tolist() == [0, 0]
    assert prepared_test["StateHoliday"].tolist() == ["0", "0"]
    assert prepared_train["Customers"].tolist() == [1]
    assert prepared_train_merged["Customers"].tolist() == [1]
    assert "StoreType" in prepared_train_merged.columns
    assert "StoreType" in prepared_test_merged.columns
    assert prepared_train_merged.loc[0, "State"] == "TS"
    assert prepared_train_merged.loc[0, "WeatherRain"] == 1
    assert prepared_train_merged.loc[0, "WeatherFog"] == 1
    assert prepared_train_merged.loc[0, "Max_Gust_SpeedKm_h"] == 12
    assert prepared_train_merged.loc[0, "WeatherMissingCount"] == 1
    assert prepared_test_merged.loc[1, "WeatherRain"] == 0
    assert set(prepared_weather["State"]) == {"TS"}
    assert report["weather_rows"] == 2
    assert report["weather_missing_values_imputed"] == 1
    assert report["train_weather_rows_unmatched"] == 0
    assert report["test_weather_rows_unmatched"] == 0
    assert prepared_store.loc[0, "CompetitionDistance"] == -1
    assert prepared_store.loc[0, "CompetitionMissing"] == 1
    assert prepared_store.loc[0, "Promo2"] == 0
    assert prepared_store.loc[0, "PromoInterval"] == ""
    assert report["missing_test_open_original"] == 2
    assert (raw_dir / "test.csv").read_bytes() == raw_test_before
