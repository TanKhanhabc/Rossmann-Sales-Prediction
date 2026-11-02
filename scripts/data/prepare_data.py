"""Clean source CSVs and write model-ready datasets to data/processed/."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _clean_schedule(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    _require_columns(frame, {"Date", "StateHoliday", "Promo", "SchoolHoliday"}, name)
    cleaned = frame.copy()
    cleaned["Date"] = pd.to_datetime(cleaned["Date"], errors="raise")
    cleaned["StateHoliday"] = (
        cleaned["StateHoliday"].fillna("0").astype(str).str.strip().replace({"0.0": "0"})
    )
    for column in ("Promo", "SchoolHoliday"):
        cleaned[column] = (
            pd.to_numeric(cleaned[column], errors="coerce").fillna(0).astype("int8")
        )
    return cleaned


def _clean_stores(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        frame,
        {
            "Store", "CompetitionDistance", "Promo2", "PromoInterval",
            "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
            "Promo2SinceWeek", "Promo2SinceYear",
        },
        "store.csv",
    )
    cleaned = frame.copy()
    distance = pd.to_numeric(cleaned["CompetitionDistance"], errors="coerce")
    cleaned["CompetitionMissing"] = distance.isna().astype("int8")
    cleaned["CompetitionDistance"] = distance.fillna(-1).clip(-1, 100000)
    cleaned["Promo2"] = (
        pd.to_numeric(cleaned["Promo2"], errors="coerce").fillna(0).astype("int8")
    )
    cleaned["PromoInterval"] = cleaned["PromoInterval"].fillna("").astype(str).str.strip()
    for column in (
        "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
        "Promo2SinceWeek", "Promo2SinceYear",
    ):
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned


def prepare_data(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
) -> dict[str, int]:
    """Prepare clean datasets without modifying the source CSVs."""
    for name in ("train.csv", "test.csv", "store.csv"):
        if not (raw_dir / name).is_file():
            raise FileNotFoundError(f"Missing {raw_dir / name}. Add the source CSV to data/raw/.")

    train = pd.read_csv(raw_dir / "train.csv", low_memory=False)
    test = pd.read_csv(raw_dir / "test.csv", low_memory=False)
    stores = pd.read_csv(raw_dir / "store.csv", low_memory=False)
    _require_columns(test, {"Open", "DayOfWeek"}, "test.csv")
    test_open_missing = int(test["Open"].isna().sum())

    train = _clean_schedule(train, "train.csv")
    test = _clean_schedule(test, "test.csv")
    test["Open"] = test["Open"].fillna(test["DayOfWeek"].ne(7).astype(int)).astype("int8")
    stores = _clean_stores(stores)

    processed_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(processed_dir / "train.csv", index=False)
    test.to_csv(processed_dir / "test.csv", index=False)
    stores.to_csv(processed_dir / "store.csv", index=False)
    report = {
        "train_rows": len(train),
        "test_rows": len(test),
        "store_rows": len(stores),
        "missing_test_open_original": test_open_missing,
    }
    (processed_dir / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    result = prepare_data()
    print(
        "Prepared datasets in "
        f"{PROCESSED_DIR} ({result['train_rows']:,} train, "
        f"{result['test_rows']:,} test, {result['store_rows']:,} stores; "
        f"{result['missing_test_open_original']} missing test Open values filled)."
    )
