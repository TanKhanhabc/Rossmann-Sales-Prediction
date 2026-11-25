"""Centralized access to datasets prepared by scripts/preprocessing/prepare_data.py."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed"
EXTERNAL_PROCESSED_DIR = PROCESSED_DIR / "external"
MERGED_DIR = ROOT / "data" / "merged"
MODEL_FILE = ROOT / "models" / "rossmann_sales_lgbm.joblib"
METRICS_FILE = ROOT / "results" / "evaluation_metrics.json"
REPORTS_DIR = ROOT / "reports"


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
def get_train_summary() -> pd.DataFrame:
    return pd.read_csv(
        require_csv("train_merged.csv", MERGED_DIR),
        usecols=["Date", "Store", "Sales", "Open"],
        parse_dates=["Date"],
        low_memory=False,
    )


@lru_cache(maxsize=1)
def get_test() -> pd.DataFrame:
    return pd.read_csv(
        require_csv("test_merged.csv", MERGED_DIR), parse_dates=["Date"], low_memory=False
    )


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
def get_preprocessing_report() -> dict:
    path = require_csv("preprocessing_report.json")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def overview() -> dict:
    df = get_train_summary()
    stores = get_stores()
    daily = df.groupby("Date", sort=True)["Sales"].sum()
    top = df.groupby("Store", sort=False)["Sales"].sum().nlargest(8)
    return {
        "train_rows": len(df), "test_rows": len(get_test()),
        "stores_total": len(stores), "stores_in_train": int(df.Store.nunique()),
        "stores_in_test": int(get_test().Store.nunique()),
        "date_start": df.Date.min().date().isoformat(),
        "date_end": df.Date.max().date().isoformat(),
        "test_start": get_test().Date.min().date().isoformat(),
        "test_end": get_test().Date.max().date().isoformat(),
        "total_sales": int(df.Sales.sum()),
        "mean_sales_open": round(float(df.loc[df.Open.eq(1), "Sales"].mean()), 2),
        "open_days_records": int(df.Open.sum()),
        "daily_average_revenue": round(float(daily.mean()), 2),
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
    for chunk in pd.read_csv(train_path, chunksize=100_000, low_memory=False):
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
