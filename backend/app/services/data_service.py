"""Centralized access to datasets prepared by scripts/data/prepare_data.py."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_FILE = ROOT / "models" / "rossmann_sales_lgbm.joblib"
METRICS_FILE = ROOT / "results" / "evaluation_metrics.json"
SUBMISSION_FILE = ROOT / "results" / "submission.csv"


def require_csv(name: str) -> Path:
    path = PROCESSED_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing prepared dataset: {path}. "
            "Run `python scripts/data/prepare_data.py` from the project root."
        )
    return path


@lru_cache(maxsize=1)
def get_stores() -> pd.DataFrame:
    return pd.read_csv(require_csv("store.csv"))


@lru_cache(maxsize=1)
def get_train() -> pd.DataFrame:
    return pd.read_csv(require_csv("train.csv"), parse_dates=["Date"], low_memory=False)


@lru_cache(maxsize=1)
def get_test() -> pd.DataFrame:
    return pd.read_csv(require_csv("test.csv"), parse_dates=["Date"], low_memory=False)


@lru_cache(maxsize=1)
def get_preprocessing_report() -> dict:
    path = require_csv("preprocessing_report.json")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def overview() -> dict:
    df = get_train()
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


def get_history(store: int | None = None, days: int = 45) -> list[dict]:
    df = get_train()
    if store is not None:
        df = df[df.Store.eq(store)]
    daily = df.groupby("Date", sort=True).agg(sales=("Sales", "sum"), customers=("Customers", "sum"))
    daily = daily.tail(days)
    return [{"date": idx.date().isoformat(), "sales": int(row.sales), "customers": int(row.customers)}
            for idx, row in daily.iterrows()]


def data_profile() -> dict:
    return {
        "train": {"rows": len(get_train()), "columns": list(get_train().columns),
                  "missing": {k: int(v) for k, v in get_train().isna().sum().items() if v}},
        "test": {"rows": len(get_test()), "columns": list(get_test().columns),
                 "missing_open_original": get_preprocessing_report()["missing_test_open_original"]},
        "store": {"rows": len(get_stores()), "columns": list(get_stores().columns),
                  "missing": {k: int(v) for k, v in get_stores().isna().sum().items() if v}},
        "notes": ["Customers and Sales only exist in train.csv, never prediction features.",
                  "Missing test Open values are estimated from day of week during data preparation.",
                  "Missing CompetitionDistance values are represented by -1 and flagged.",
                  "All dates are historical (2013–2015); not a live feed."],
    }
