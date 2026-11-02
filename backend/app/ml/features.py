"""Point-in-time feature engineering. NEVER use Sales or Customers as predictors.

All features are known/derivable at prediction time from the schedule and store.csv.
"""
from __future__ import annotations
import calendar
import numpy as np
import pandas as pd

CATEGORICAL = ["Store", "StateHoliday", "StoreType", "Assortment"]
FEATURES = [
    "Store", "DayOfWeek", "Promo", "SchoolHoliday", "StateHoliday",
    "StoreType", "Assortment", "CompetitionDistance", "CompetitionMissing",
    "CompetitionOpenMonths", "Promo2", "Promo2Active", "Promo2MonthActive",
    "Month", "Day", "Year", "WeekOfYear", "DayOfYear", "Quarter",
    "IsWeekend", "IsMonthStart", "IsMonthEnd", "MonthSin", "MonthCos",
    "DayOfYearSin", "DayOfYearCos", "Promo2WeeksElapsed",
]


def make_features(schedule: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    required = {"Store", "Date", "Promo", "SchoolHoliday", "StateHoliday"}
    missing = required.difference(schedule.columns)
    if missing:
        raise ValueError(f"Missing schedule columns: {sorted(missing)}")
    if stores["Store"].duplicated().any():
        raise ValueError("store.csv contains duplicate Store keys")
    df = schedule.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    store_fields = [c for c in stores.columns if c != "Store"]
    # Drop stale store columns if caller passed an already-joined dataset.
    df = df.drop(columns=[x for x in store_fields if x in df.columns], errors="ignore")
    df = df.merge(stores, on="Store", how="left", validate="many_to_one", sort=False)
    if df["StoreType"].isna().any():
        raise ValueError("Some Store IDs do not exist in store.csv")
    dt = df["Date"].dt
    df["DayOfWeek"] = dt.dayofweek + 1
    df["Month"] = dt.month
    df["Day"] = dt.day
    df["Year"] = dt.year
    df["WeekOfYear"] = dt.isocalendar().week.astype("int16")
    df["DayOfYear"] = dt.dayofyear
    df["Quarter"] = dt.quarter
    df["IsWeekend"] = df["DayOfWeek"].ge(6).astype("int8")
    df["IsMonthStart"] = dt.is_month_start.astype("int8")
    df["IsMonthEnd"] = dt.is_month_end.astype("int8")
    df["MonthSin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["MonthCos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["DayOfYearSin"] = np.sin(2 * np.pi * df["DayOfYear"] / 365.25)
    df["DayOfYearCos"] = np.cos(2 * np.pi * df["DayOfYear"] / 365.25)
    df["CompetitionMissing"] = df["CompetitionMissing"].astype("int8")
    df["CompetitionDistance"] = df["CompetitionDistance"].clip(-1, 100000)
    comp_y = pd.to_numeric(df["CompetitionOpenSinceYear"], errors="coerce")
    comp_m = pd.to_numeric(df["CompetitionOpenSinceMonth"], errors="coerce")
    comp_age = (df["Year"] - comp_y) * 12 + (df["Month"] - comp_m)
    df["CompetitionOpenMonths"] = comp_age.clip(lower=0).fillna(-1)
    p2_year = pd.to_numeric(df["Promo2SinceYear"], errors="coerce")
    p2_week = pd.to_numeric(df["Promo2SinceWeek"], errors="coerce")
    weeks = (df["Year"] - p2_year) * 52 + df["WeekOfYear"] - p2_week
    p2 = df["Promo2"].astype("int8")
    df["Promo2"] = p2
    df["Promo2Active"] = (p2.eq(1) & weeks.ge(0)).astype("int8")
    df["Promo2WeeksElapsed"] = weeks.where(p2.eq(1) & weeks.ge(0), -1).fillna(-1)
    month_abbr = df["Month"].map(dict(enumerate(calendar.month_abbr)))
    interval = df["PromoInterval"].astype(str)
    unique = interval.unique()
    interval_set = {k: set(k.split(",")) for k in unique}
    pairs = pd.Series(list(zip(interval.to_numpy(), month_abbr.to_numpy())), index=df.index)
    # Small distinct set of (promo interval, month) pairs (not a million row Python loop).
    mapping = {(i, m): int(m in interval_set[i]) for i in unique for m in calendar.month_abbr[1:]}
    df["Promo2MonthActive"] = (pairs.map(mapping).fillna(0).astype("int8") * df["Promo2Active"])
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    return df[FEATURES]
