from __future__ import annotations
from datetime import date, timedelta
from typing import Literal
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from app.model.predict_model import infer
from app.services.data_service import (
    get_stores,
    get_train_for_store,
    get_test_for_store,
    get_test_schedule,
    get_test_store_ranges,
)

router = APIRouter(prefix="/api", tags=["Forecast & Prediction"])

class PredictRequest(BaseModel):
    store: int = Field(ge=1)
    date: date
    promo: int = Field(default=0, ge=0, le=1)
    open: int = Field(default=1, ge=0, le=1)
    state_holiday: Literal["0", "a", "b", "c"] = "0"
    school_holiday: int = Field(default=0, ge=0, le=1)


class ForecastSchedule(BaseModel):
    date: date
    promo: int = Field(ge=0, le=1)
    open: int = Field(ge=0, le=1)
    school_holiday: int = Field(ge=0, le=1)
    state_holiday: Literal["0", "a", "b", "c"] = "0"


class ForecastRequest(BaseModel):
    store: int = Field(ge=1)
    start_date: date
    days: int = Field(ge=1, le=30)
    schedule: list[ForecastSchedule] = Field(default_factory=list)
    assumed_promo: int = Field(default=0, ge=0, le=1)
    assumed_school_holiday: int = Field(default=0, ge=0, le=1)
    assumed_state_holiday: Literal["0", "a", "b", "c"] = "0"


def validate_store(store: int) -> None:
    if store not in set(get_stores().Store):
        raise HTTPException(status_code=404, detail=f"Store {store} không tồn tại")


def to_frame(requests: list[PredictRequest]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Store": x.store, "Date": pd.Timestamp(x.date), "Promo": x.promo,
        "Open": x.open, "StateHoliday": x.state_holiday,
        "SchoolHoliday": x.school_holiday,
    } for x in requests])


def make_predictions(requests: list[PredictRequest]) -> list[dict]:
    for item in requests:
        validate_store(item.store)
    try:
        frames = []
        actual_sales = []
        conditions_match = []
        out_of_range = []
        historical_stores = {
            item.store
            for item in requests
            if date(2013, 1, 1) <= item.date <= date(2015, 7, 31)
        }
        test_stores = {
            item.store
            for item in requests
            if date(2015, 8, 1) <= item.date <= date(2015, 9, 17)
        }
        train_by_store = {
            store: get_train_for_store(store) for store in historical_stores
        }
        test_by_store = {
            store: get_test_for_store(store) for store in test_stores
        }
        for item in requests:
            frame = to_frame([item])
            timestamp = pd.Timestamp(item.date)
            if date(2013, 1, 1) <= item.date <= date(2015, 7, 31):
                out_of_range.append(False)
                train = train_by_store[item.store]
                row = train.loc[
                    train.Date.eq(timestamp)
                ]
                if not row.empty:
                    original = row.iloc[0]
                    historical = original.copy()
                    historical["Promo"] = item.promo
                    historical["Open"] = item.open
                    historical["SchoolHoliday"] = item.school_holiday
                    historical["StateHoliday"] = item.state_holiday
                    frames.append(pd.DataFrame([historical]))
                    actual_match = (
                        int(original.Promo) == item.promo
                        and int(original.Open) == item.open
                        and int(original.SchoolHoliday) == item.school_holiday
                        and str(original.StateHoliday).replace("0.0", "0") == item.state_holiday
                    )
                    conditions_match.append(actual_match)
                    actual_sales.append(
                        float(original.Sales) if actual_match else None
                    )
                else:
                    frames.append(frame)
                    conditions_match.append(False)
                    actual_sales.append(None)
            else:
                if date(2015, 8, 1) <= item.date <= date(2015, 9, 17):
                    test = test_by_store[item.store]
                    row = test.loc[
                        test.Date.eq(timestamp)
                    ]
                    if not row.empty:
                        out_of_range.append(False)
                        historical = row.iloc[0].copy()
                        historical["Promo"] = item.promo
                        historical["Open"] = item.open
                        historical["SchoolHoliday"] = item.school_holiday
                        historical["StateHoliday"] = item.state_holiday
                        frames.append(pd.DataFrame([historical]))
                    else:
                        out_of_range.append(True)
                        frames.append(frame)
                else:
                    out_of_range.append(True)
                    frames.append(frame)
                conditions_match.append(False)
                actual_sales.append(None)

        frame = pd.concat(frames, ignore_index=True)
        values = infer(frame)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    results = []
    for item, value, actual, match, outside in zip(
        requests, values, actual_sales, conditions_match, out_of_range
    ):
        if item.open == 0:
            predicted = 0.0
        else:
            predicted = float(value)
        results.append({
            "store": item.store,
            "date": item.date.isoformat(),
            "predicted_sales": round(predicted, 2),
            "open": item.open,
            "promo": item.promo,
            "state_holiday": item.state_holiday,
            "school_holiday": item.school_holiday,
            "actual_sales": round(actual, 2) if actual is not None else None,
            "conditions_match_actual": match,
            "is_out_of_range": outside,
            "currency": "EUR",
        })
    return results


@router.post("/predict")
def predict(request: PredictRequest):
    return make_predictions([request])[0]


@router.get("/test-predictions")
def test_predictions(
    store: int = Query(..., ge=1),
    start_date: date = Query(...),
    end_date: date = Query(...),
):
    validate_store(store)
    if start_date > end_date:
        raise HTTPException(
            status_code=422,
            detail="Ngày bắt đầu phải trước hoặc bằng ngày kết thúc",
        )

    ranges = get_test_store_ranges()
    store_range = ranges.loc[ranges.Store.eq(store)]
    if store_range.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Store {store} không có dữ liệu trong test.csv",
        )
    test_start = store_range.iloc[0].start_date.date()
    test_end = store_range.iloc[0].end_date.date()
    if start_date < test_start or end_date > test_end:
        raise HTTPException(
            status_code=422,
            detail=f"Khoảng ngày của cửa hàng {store} phải nằm trong test.csv ({test_start} – {test_end})",
        )

    test = get_test_schedule()
    selected = test.loc[
        test.Store.eq(store)
        & test.Date.ge(pd.Timestamp(start_date))
        & test.Date.le(pd.Timestamp(end_date))
    ].sort_values("Date")
    if selected.empty:
        values = []
    else:
        try:
            values = infer(selected)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    points = [
        {
            "date": row.Date.date().isoformat(),
            "predicted_sales": round(float(value), 2),
            "open": int(row.Open),
            "promo": int(row.Promo),
            "state_holiday": str(row.StateHoliday).replace("0.0", "0"),
            "school_holiday": int(row.SchoolHoliday),
        }
        for row, value in zip(selected.itertuples(index=False), values)
    ]
    return {
        "store": store,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "total_predicted_sales": round(sum(point["predicted_sales"] for point in points), 2),
        "points": points,
    }


@router.get("/forecast")
def forecast(
    store: int = Query(1, ge=1), start_date: date = Query(date(2015, 8, 1)),
    days: int = Query(14, ge=1, le=30),
    assumed_promo: int = Query(0, ge=0, le=1),
    assumed_school_holiday: int = Query(0, ge=0, le=1),
    assumed_state_holiday: Literal["0", "a", "b", "c"] = "0",
):
    return make_forecast(
        store, start_date, days, [], assumed_promo,
        assumed_school_holiday, assumed_state_holiday,
    )


@router.post("/forecast")
def forecast_with_schedule(request: ForecastRequest):
    return make_forecast(
        request.store, request.start_date, request.days, request.schedule,
        request.assumed_promo, request.assumed_school_holiday,
        request.assumed_state_holiday,
    )


def make_forecast(
    store: int,
    start_date: date,
    days: int,
    user_schedule: list[ForecastSchedule],
    assumed_promo: int,
    assumed_school_holiday: int,
    assumed_state_holiday: str,
):
    validate_store(store)
    known = get_test_schedule()
    target_dates = pd.date_range(start_date, periods=days, freq="D")
    window = known.loc[known.Store.eq(store) & known.Date.isin(target_dates)]
    schedule = {row.Date.date(): row for row in window.itertuples(index=False)}
    user_schedule_by_date = {}
    for entry in user_schedule:
        if entry.date < start_date or entry.date >= start_date + timedelta(days=days):
            raise HTTPException(
                status_code=422,
                detail=f"Schedule date {entry.date.isoformat()} is outside the forecast range",
            )
        if entry.date in user_schedule_by_date:
            raise HTTPException(
                status_code=422,
                detail=f"Duplicate schedule date: {entry.date.isoformat()}",
            )
        user_schedule_by_date[entry.date] = entry
    requests = []
    sources = []
    for dt in target_dates:
        d = dt.date()
        if d in user_schedule_by_date:
            row = user_schedule_by_date[d]
            requests.append(PredictRequest(
                store=store, date=d, promo=row.promo, open=row.open,
                school_holiday=row.school_holiday, state_holiday=row.state_holiday,
            ))
            sources.append("user")
        elif d in schedule:
            row = schedule[d]
            requests.append(PredictRequest(store=store, date=d, promo=int(row.Promo), open=int(row.Open),
                                           school_holiday=int(row.SchoolHoliday),
                                           state_holiday=str(row.StateHoliday).replace("0.0", "0")))
            sources.append("test.csv")
        else:
            # Beyond given schedule: scenario assumption, NOT known future facts.
            requests.append(PredictRequest(store=store, date=d, promo=assumed_promo,
                                           open=int(d.isoweekday() != 7),
                                           school_holiday=assumed_school_holiday,
                                           state_holiday=assumed_state_holiday))
            sources.append("assumption")
    prediction = make_predictions(requests)
    for row, src, item in zip(prediction, sources, requests):
        row["schedule_source"] = src
        row["school_holiday"] = item.school_holiday
        row["state_holiday"] = item.state_holiday
    return {
        "store": store, "start_date": start_date.isoformat(), "days": days,
        "total_predicted_sales": round(float(sum(x["predicted_sales"] for x in prediction)), 2),
        "known_schedule_days": sources.count("test.csv") + sources.count("user"),
        "assumed_schedule_days": sources.count("assumption"),
        "user_schedule_days": sources.count("user"),
        "warning": "Kết quả là ước tính. Lịch người dùng được ưu tiên, ngày chưa cấu hình dùng lịch test.csv nếu có; các ngày còn lại giả định nghỉ Chủ nhật và không khuyến mãi/ngày lễ. Dữ liệu lịch sử 2013–2015, không phải dự báo thời gian thực.",
        "points": prediction,
    }
