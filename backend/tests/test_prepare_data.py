import pandas as pd

from scripts.data.prepare_data import prepare_data


def test_prepare_data_cleans_inputs_without_modifying_raw_files(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
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
    raw_test_before = (raw_dir / "test.csv").read_bytes()

    report = prepare_data(raw_dir, processed_dir)

    prepared_test = pd.read_csv(processed_dir / "test.csv", dtype={"StateHoliday": str})
    prepared_store = pd.read_csv(processed_dir / "store.csv", keep_default_na=False)
    assert prepared_test["Open"].tolist() == [1, 0]
    assert prepared_test["Promo"].tolist() == [0, 0]
    assert prepared_test["StateHoliday"].tolist() == ["0", "0"]
    assert prepared_store.loc[0, "CompetitionDistance"] == -1
    assert prepared_store.loc[0, "CompetitionMissing"] == 1
    assert prepared_store.loc[0, "Promo2"] == 0
    assert prepared_store.loc[0, "PromoInterval"] == ""
    assert report["missing_test_open_original"] == 2
    assert (raw_dir / "test.csv").read_bytes() == raw_test_before
