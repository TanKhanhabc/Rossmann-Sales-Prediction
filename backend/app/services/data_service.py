"""Centralized access to datasets prepared by scripts/preprocessing/prepare_data.py."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from threading import Lock
import json

import pandas as pd

from app.model.features import WEATHER_FEATURES

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed"
EXTERNAL_PROCESSED_DIR = PROCESSED_DIR / "external"
MERGED_DIR = ROOT / "data" / "merged"
MODEL_FILE = ROOT / "models" / "rossmann_sales_lgbm.joblib"
METRICS_FILE = ROOT / "results" / "evaluation_metrics.json"
REPORTS_DIR = ROOT / "reports"
PREDICTION_CHUNK_SIZE = 25_000
PREDICTION_CACHE_SIZE = 8
PREDICTION_DATA_LOCK = Lock()
TRAIN_PREDICTION_COLUMNS = [
    "Store", "Date", "Sales", "Promo", "Open", "SchoolHoliday", "StateHoliday",
    *WEATHER_FEATURES,
]
TEST_PREDICTION_COLUMNS = [
    "Store", "Date", "Promo", "Open", "SchoolHoliday", "StateHoliday",
    *WEATHER_FEATURES,
]
PREDICTION_SMALL_INTEGER_COLUMNS = {
    "Store": "uint16",
    "Sales": "int32",
    "Promo": "int8",
    "Open": "int8",
    "SchoolHoliday": "int8",
    "WeatherMissingCount": "int8",
    **{
        f"Weather{event}": "int8"
        for event in ("Rain", "Snow", "Fog", "Hail", "Thunderstorm")
    },
}


def require_csv(name: str, directory: Path = PROCESSED_DIR) -> Path:
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing prepared dataset: {path}. "
            "Run `python scripts/preprocessing/prepare_data.py` from the project root."
        )
    return path


@lru_cache(maxsize=1)
def get_stores() -> pd.DataFrame:
    return pd.read_csv(require_csv("store.csv"))


@lru_cache(maxsize=1)
def get_weather() -> pd.DataFrame:
    return pd.read_csv(
        require_csv("weather.csv", EXTERNAL_PROCESSED_DIR),
        parse_dates=["Date"],
        low_memory=False,
    )


@lru_cache(maxsize=1)
def get_state_names() -> pd.DataFrame:
    return pd.read_csv(require_csv("state_names.csv", EXTERNAL_PROCESSED_DIR))


@lru_cache(maxsize=1)
def get_store_states() -> pd.DataFrame:
    return pd.read_csv(require_csv("store_states.csv", EXTERNAL_PROCESSED_DIR))


@lru_cache(maxsize=1)
def get_train() -> pd.DataFrame:
    return pd.read_csv(
        require_csv("train_merged.csv", MERGED_DIR), parse_dates=["Date"], low_memory=False
    )


@lru_cache(maxsize=1)
def get_test() -> pd.DataFrame:
    return pd.read_csv(
        require_csv("test_merged.csv", MERGED_DIR), parse_dates=["Date"], low_memory=False
    )


def _read_prediction_rows(path: Path, store: int, columns: list[str]) -> pd.DataFrame:
    dtypes = {
        column: dtype
        for column, dtype in PREDICTION_SMALL_INTEGER_COLUMNS.items()
        if column in columns
    }
    dtypes["StateHoliday"] = "str"
    matches = []
    for chunk in pd.read_csv(
        path,
        usecols=columns,
        dtype=dtypes,
        parse_dates=["Date"],
        chunksize=PREDICTION_CHUNK_SIZE,
        low_memory=False,
    ):
        selected = chunk.loc[chunk["Store"].eq(store)]
        if not selected.empty:
            matches.append(selected)

    if matches:
        return pd.concat(matches, ignore_index=True)
    return pd.DataFrame(columns=columns)


@lru_cache(maxsize=PREDICTION_CACHE_SIZE)
def _load_train_for_store(store: int) -> pd.DataFrame:
    return _read_prediction_rows(
        require_csv("train_merged.csv", MERGED_DIR),
        store,
        TRAIN_PREDICTION_COLUMNS,
    )


@lru_cache(maxsize=PREDICTION_CACHE_SIZE)
def _load_test_for_store(store: int) -> pd.DataFrame:
    return _read_prediction_rows(
        require_csv("test_merged.csv", MERGED_DIR),
        store,
        TEST_PREDICTION_COLUMNS,
    )


def get_train_for_store(store: int) -> pd.DataFrame:
    with PREDICTION_DATA_LOCK:
        return _load_train_for_store(store)


def get_test_for_store(store: int) -> pd.DataFrame:
    with PREDICTION_DATA_LOCK:
        return _load_test_for_store(store)


@lru_cache(maxsize=1)
def get_test_schedule() -> pd.DataFrame:
    return pd.read_csv(require_csv("test.csv"), parse_dates=["Date"], low_memory=False)


def get_test_store_ranges() -> pd.DataFrame:
    return (
        get_test_schedule()
        .groupby("Store", as_index=False)["Date"]
        .agg(start_date="min", end_date="max")
    )


@lru_cache(maxsize=1)
def get_test_summary() -> pd.DataFrame:
    return pd.read_csv(
        require_csv("test_merged.csv", MERGED_DIR),
        usecols=["Date", "Store"],
        parse_dates=["Date"],
        dtype={"Store": "uint16"},
    )


@lru_cache(maxsize=1)
def get_preprocessing_report() -> dict:
    path = require_csv("preprocessing_report.json")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def overview() -> dict:
    train_path = require_csv("train_merged.csv", MERGED_DIR)
    test = get_test_summary()
    stores = get_stores()
    train_rows = 0
    train_stores = set()
    sales_by_store = {}
    sales_by_date = {}
    total_sales = 0
    open_sales = 0
    open_records = 0
    open_days_records = 0
    date_start = None
    date_end = None
    for chunk in pd.read_csv(
        train_path,
        usecols=["Date", "Store", "Sales", "Open"],
        parse_dates=["Date"],
        dtype={"Store": "uint16", "Sales": "int32", "Open": "int8"},
        chunksize=PREDICTION_CHUNK_SIZE,
    ):
        train_rows += len(chunk)
        total_sales += int(chunk["Sales"].sum())
        is_open = chunk["Open"].eq(1)
        open_sales += int(chunk.loc[is_open, "Sales"].sum())
        open_records += int(is_open.sum())
        open_days_records += int(chunk["Open"].sum())
        train_stores.update(int(store) for store in chunk["Store"].unique())
        chunk_start = chunk["Date"].min()
        chunk_end = chunk["Date"].max()
        date_start = chunk_start if date_start is None else min(date_start, chunk_start)
        date_end = chunk_end if date_end is None else max(date_end, chunk_end)
        for store, sales in chunk.groupby("Store", sort=False)["Sales"].sum().items():
            store_id = int(store)
            sales_by_store[store_id] = sales_by_store.get(store_id, 0) + int(sales)
        for day, sales in chunk.groupby("Date", sort=False)["Sales"].sum().items():
            sales_by_date[day] = sales_by_date.get(day, 0) + int(sales)

    top = pd.Series(sales_by_store, dtype="int64").nlargest(8)
    return {
        "train_rows": train_rows, "test_rows": len(test),
        "stores_total": len(stores), "stores_in_train": len(train_stores),
        "stores_in_test": int(test.Store.nunique()),
        "date_start": date_start.date().isoformat(),
        "date_end": date_end.date().isoformat(),
        "test_start": test.Date.min().date().isoformat(),
        "test_end": test.Date.max().date().isoformat(),
        "total_sales": total_sales,
        "mean_sales_open": round(open_sales / open_records, 2),
        "open_days_records": open_days_records,
        "daily_average_revenue": round(sum(sales_by_date.values()) / len(sales_by_date), 2),
        "top_stores": [{"store": int(s), "sales": int(v)} for s, v in top.items()],
        "missing_test_open": int(get_preprocessing_report()["missing_test_open_original"]),
    }



def data_profile() -> dict:
    train_path = require_csv("train_merged.csv", MERGED_DIR)
    test_path = require_csv("test_merged.csv", MERGED_DIR)
    report = get_preprocessing_report()
    train_columns = list(pd.read_csv(train_path, nrows=0).columns)
    test_columns = list(pd.read_csv(test_path, nrows=0).columns)
    missing = pd.Series(dtype="int64")
    for chunk in pd.read_csv(
        train_path,
        chunksize=PREDICTION_CHUNK_SIZE,
        dtype={"StateHoliday": "str"},
    ):
        chunk_missing = chunk.isna().sum()
        missing = missing.add(chunk_missing, fill_value=0)

    return {
        "train": {"rows": int(report["train_rows"]), "columns": train_columns,
                 "missing": {k: int(v) for k, v in missing.items() if v}},
        "test": {"rows": int(report["test_rows"]), "columns": test_columns,
                 "missing_open_original": report["missing_test_open_original"]},
        "store": {"rows": len(get_stores()), "columns": list(get_stores().columns),
                  "missing": {k: int(v) for k, v in get_stores().isna().sum().items() if v}},
        "weather": {
            "rows": int(report["weather_rows"]),
            "columns": list(pd.read_csv(
                require_csv("weather.csv", EXTERNAL_PROCESSED_DIR),
                nrows=0,
            ).columns),
            "imputed_measurements": int(report["weather_missing_values_imputed"]),
        },
        "state_names": {"rows": len(get_state_names()), "columns": list(get_state_names().columns)},
        "store_states": {"rows": len(get_store_states()), "columns": list(get_store_states().columns)},
        "notes": ["Sales only exists in train data and is never a prediction feature.",
                  "Missing test Open values are estimated from day of week during data preparation.",
                  "Missing CompetitionDistance values are represented by -1 and flagged.",
                  "Weather observations are mapped by store state and date; unknown dates use historical state medians.",
                  "All dates are historical (2013–2015); not a live feed."],
    }
