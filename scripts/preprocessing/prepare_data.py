"""Clean source CSVs and write prepared and merged datasets under data/."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MERGED_DIR = ROOT / "data" / "merged"
EXTERNAL_DIR = ROOT / "data" / "external"

WEATHER_NUMERIC_COLUMNS = [
    "Max_TemperatureC", "Mean_TemperatureC", "Min_TemperatureC",
    "Dew_PointC", "MeanDew_PointC", "Min_DewpointC",
    "Max_Humidity", "Mean_Humidity", "Min_Humidity",
    "Max_Sea_Level_PressurehPa", "Mean_Sea_Level_PressurehPa",
    "Min_Sea_Level_PressurehPa", "Max_VisibilityKm", "Mean_VisibilityKm",
    "Min_VisibilitykM", "Max_Wind_SpeedKm_h", "Mean_Wind_SpeedKm_h",
    "Max_Gust_SpeedKm_h", "Precipitationmm", "CloudCover", "WindDirDegrees",
]
WEATHER_EVENT_TYPES = ["Rain", "Snow", "Fog", "Hail", "Thunderstorm"]
WEATHER_FEATURE_COLUMNS = [
    *[column for column in WEATHER_NUMERIC_COLUMNS if column != "WindDirDegrees"],
    "WindDirSin", "WindDirCos", "WeatherMissingCount",
    *[f"Weather{event}" for event in WEATHER_EVENT_TYPES],
]


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


def _clean_state_names(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, {"StateName", "State"}, "state_names.csv")
    cleaned = frame[["StateName", "State"]].copy()
    for column in ("StateName", "State"):
        cleaned[column] = cleaned[column].astype("string").str.strip()
    if cleaned.isna().any().any() or cleaned.duplicated("State").any():
        raise ValueError("state_names.csv contains missing values or duplicate state codes")
    if cleaned["StateName"].duplicated().any():
        raise ValueError("state_names.csv contains duplicate state names")
    return cleaned


def _clean_store_states(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, {"Store", "State"}, "store_states.csv")
    cleaned = frame[["Store", "State"]].copy()
    cleaned["Store"] = pd.to_numeric(cleaned["Store"], errors="raise").astype("int64")
    cleaned["State"] = cleaned["State"].astype("string").str.strip()
    if cleaned["State"].isna().any() or cleaned["Store"].duplicated().any():
        raise ValueError("store_states.csv contains missing states or duplicate Store keys")
    return cleaned


def _clean_weather(frame: pd.DataFrame, state_names: pd.DataFrame) -> pd.DataFrame:
    required = {"file", "Date", "Events", *WEATHER_NUMERIC_COLUMNS}
    _require_columns(frame, required, "weather.csv")
    cleaned = frame.rename(columns={"file": "StateName"}).copy()
    cleaned["StateName"] = cleaned["StateName"].astype("string").str.strip()
    cleaned["Date"] = pd.to_datetime(cleaned["Date"], errors="raise")
    cleaned = cleaned.merge(
        state_names, on="StateName", how="left", validate="many_to_one", sort=False
    )
    if cleaned["State"].isna().any():
        unknown = sorted(cleaned.loc[cleaned["State"].isna(), "StateName"].unique())
        raise ValueError(f"weather.csv contains unmapped state names: {unknown}")
    if cleaned.duplicated(["State", "Date"]).any():
        raise ValueError("weather.csv contains duplicate state/date rows")

    for column in WEATHER_NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned["WeatherMissingCount"] = cleaned[WEATHER_NUMERIC_COLUMNS].isna().sum(axis=1)
    state_medians = cleaned.groupby("State", observed=True)[WEATHER_NUMERIC_COLUMNS].transform("median")
    cleaned[WEATHER_NUMERIC_COLUMNS] = cleaned[WEATHER_NUMERIC_COLUMNS].fillna(state_medians)
    global_medians = cleaned[WEATHER_NUMERIC_COLUMNS].median()
    if global_medians.isna().any():
        missing = global_medians[global_medians.isna()].index.tolist()
        raise ValueError(f"weather.csv has no usable values for columns: {missing}")
    cleaned[WEATHER_NUMERIC_COLUMNS] = cleaned[WEATHER_NUMERIC_COLUMNS].fillna(global_medians)

    events = cleaned["Events"].fillna("").astype(str).str.strip()
    for event in WEATHER_EVENT_TYPES:
        cleaned[f"Weather{event}"] = (
            events.str.split("-").map(lambda values: int(event in values)).astype("int8")
        )
    direction = np.deg2rad(cleaned["WindDirDegrees"])
    cleaned["WindDirSin"] = np.sin(direction)
    cleaned["WindDirCos"] = np.cos(direction)
    return cleaned[
        ["State", "Date", *WEATHER_NUMERIC_COLUMNS, "WeatherMissingCount",
         "WindDirSin", "WindDirCos", *[f"Weather{event}" for event in WEATHER_EVENT_TYPES]]
    ]


def prepare_data(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    merged_dir: Path = MERGED_DIR,
    external_dir: Path = EXTERNAL_DIR,
) -> dict[str, int]:
    """Prepare clean datasets without modifying the source CSVs."""
    for name in ("train.csv", "test.csv", "store.csv"):
        if not (raw_dir / name).is_file():
            raise FileNotFoundError(f"Missing {raw_dir / name}. Add the source CSV to data/raw/.")
    for name in ("weather.csv", "state_names.csv", "store_states.csv"):
        if not (external_dir / name).is_file():
            raise FileNotFoundError(
                f"Missing {external_dir / name}. Add the external CSV to data/external/."
            )

    train = pd.read_csv(raw_dir / "train.csv", low_memory=False)
    test = pd.read_csv(raw_dir / "test.csv", low_memory=False)
    stores = pd.read_csv(raw_dir / "store.csv", low_memory=False)
    state_names = _clean_state_names(pd.read_csv(external_dir / "state_names.csv"))
    store_states = _clean_store_states(pd.read_csv(external_dir / "store_states.csv"))
    weather = _clean_weather(
        pd.read_csv(external_dir / "weather.csv", low_memory=False), state_names
    )
    _require_columns(test, {"Open", "DayOfWeek"}, "test.csv")
    test_open_missing = int(test["Open"].isna().sum())

    train = _clean_schedule(train, "train.csv")
    test = _clean_schedule(test, "test.csv")
    test["Open"] = test["Open"].fillna(test["DayOfWeek"].ne(7).astype(int)).astype("int8")
    stores = _clean_stores(stores)
    stores = stores.merge(
        store_states, on="Store", how="left", validate="one_to_one", sort=False
    )
    if stores["State"].isna().any():
        missing_stores = stores.loc[stores["State"].isna(), "Store"].tolist()
        raise ValueError(f"store_states.csv is missing store IDs: {missing_stores[:10]}")
    unknown_states = set(stores["State"]).difference(state_names["State"])
    if unknown_states:
        raise ValueError(f"store_states.csv contains unmapped state codes: {sorted(unknown_states)}")
    if set(store_states["Store"]).difference(stores["Store"]):
        raise ValueError("store_states.csv contains Store IDs absent from store.csv")

    # Keep Customers for historical analytics; model features explicitly exclude it.
    train_merged = train.merge(stores, on="Store", how="left", validate="many_to_one", sort=False)
    test_merged = test.merge(stores, on="Store", how="left", validate="many_to_one", sort=False)
    train_merged = train_merged.merge(
        weather, on=["State", "Date"], how="left", validate="many_to_one", sort=False
    )
    test_merged = test_merged.merge(
        weather, on=["State", "Date"], how="left", validate="many_to_one", sort=False
    )
    if train_merged[WEATHER_FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Some train rows could not be matched to cleaned weather data")
    if test_merged[WEATHER_FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Some test rows could not be matched to cleaned weather data")

    processed_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)
    external_processed_dir = processed_dir / "external"
    external_processed_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(processed_dir / "train.csv", index=False)
    train_merged.to_csv(merged_dir / "train_merged.csv", index=False)
    test.to_csv(processed_dir / "test.csv", index=False)
    test_merged.to_csv(merged_dir / "test_merged.csv", index=False)
    stores.to_csv(processed_dir / "store.csv", index=False)
    state_names.to_csv(external_processed_dir / "state_names.csv", index=False)
    store_states.to_csv(external_processed_dir / "store_states.csv", index=False)
    weather.to_csv(external_processed_dir / "weather.csv", index=False)
    report = {
        "train_rows": len(train),
        "test_rows": len(test),
        "store_rows": len(stores),
        "missing_test_open_original": test_open_missing,
        "weather_rows": len(weather),
        "weather_missing_values_imputed": int(weather["WeatherMissingCount"].sum()),
        "train_weather_rows_unmatched": 0,
        "test_weather_rows_unmatched": 0,
    }
    (processed_dir / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    result = prepare_data()
    print(
        "Prepared datasets in "
        f"{PROCESSED_DIR} and merged datasets in {MERGED_DIR} "
        f"({result['train_rows']:,} train, "
        f"{result['test_rows']:,} test, {result['store_rows']:,} stores; "
        f"{result['missing_test_open_original']} missing test Open values filled; "
        f"{result['weather_rows']:,} weather rows cleaned and mapped)."
    )
