from concurrent.futures import ThreadPoolExecutor
import time

import pandas as pd

from app.model.features import WEATHER_FEATURES
from app.services import data_service


def test_prediction_reader_streams_selected_columns_and_keeps_only_requested_store(
    tmp_path, monkeypatch
):
    source = tmp_path / "train_merged.csv"
    records = []
    for store, day in ((1, "2015-07-30"), (2, "2015-07-30"), (1, "2015-07-31")):
        records.append({
            "Store": store,
            "Date": day,
            "Sales": 1250,
            "Promo": 1,
            "Open": 1,
            "SchoolHoliday": 0,
            "StateHoliday": "0",
            **{feature: 1 for feature in WEATHER_FEATURES},
        })
    pd.DataFrame(records).to_csv(source, index=False)
    original_read_csv = pd.read_csv
    reader_options = {}

    def observe_read_csv(*args, **kwargs):
        reader_options.update(kwargs)
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(data_service.pd, "read_csv", observe_read_csv)

    rows = data_service._read_prediction_rows(
        source, 1, data_service.TRAIN_PREDICTION_COLUMNS
    )

    assert rows.Store.tolist() == [1, 1]
    assert rows.Sales.tolist() == [1250, 1250]
    assert rows.StateHoliday.tolist() == ["0", "0"]
    assert str(rows.Store.dtype) == "uint16"
    assert str(rows.Sales.dtype) == "int32"
    assert reader_options["chunksize"] == data_service.PREDICTION_CHUNK_SIZE
    assert set(reader_options["usecols"]) == set(data_service.TRAIN_PREDICTION_COLUMNS)


def test_concurrent_cache_misses_load_each_store_once(monkeypatch):
    calls = []

    def load_rows(path, store, columns):
        calls.append(store)
        time.sleep(0.02)
        return pd.DataFrame({"Store": [store]})

    monkeypatch.setattr(data_service, "_read_prediction_rows", load_rows)
    data_service._load_train_for_store.cache_clear()

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(data_service.get_train_for_store, [1] * 8))
    finally:
        data_service._load_train_for_store.cache_clear()

    assert calls == [1]
    assert all(result.Store.tolist() == [1] for result in results)
