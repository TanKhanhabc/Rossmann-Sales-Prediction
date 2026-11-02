from __future__ import annotations
from datetime import date, timedelta
from functools import lru_cache
from typing import Literal
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from app.ml.model_service import infer, load_model
from app.services.data_service import SUBMISSION_FILE, get_stores, get_test

router = APIRouter(prefix="/api", tags=["Forecast & Prediction"])

class PredictRequest(BaseModel):
    store: int = Field(ge=1)
    date: date
    promo: int = Field(default=0, ge=0, le=1)
    open: int = Field(default=1, ge=0, le=1)
    state_holiday: Literal["0", "a", "b", "c"] = "0"
    school_holiday: int = Field(default=0, ge=0, le=1)


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
        values = infer(to_frame(requests))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return [{"store": item.store, "date": item.date.isoformat(),
             "predicted_sales": round(float(value), 2),
             "open": item.open, "promo": item.promo,
             "currency": "EUR"}
            for item, value in zip(requests, values)]


@router.post("/predict")
def predict(request: PredictRequest):
    return make_predictions([request])[0]


@router.post("/predict-batch")
def predict_batch(requests: list[PredictRequest]):
    if not requests or len(requests) > 500:
        raise HTTPException(status_code=422, detail="Provide between 1 and 500 requests")
    return {"count": len(requests), "predictions": make_predictions(requests)}


@router.get("/forecast")
def forecast(
    store: int = Query(1, ge=1), start_date: date = Query(date(2015, 8, 1)),
    days: int = Query(14, ge=1, le=30),
    assumed_promo: int = Query(0, ge=0, le=1),
    assumed_school_holiday: int = Query(0, ge=0, le=1),
    assumed_state_holiday: Literal["0", "a", "b", "c"] = "0",
):
    validate_store(store)
    known = get_test()
    target_dates = pd.date_range(start_date, periods=days, freq="D")
    window = known.loc[known.Store.eq(store) & known.Date.isin(target_dates)]
    schedule = {row.Date.date(): row for row in window.itertuples(index=False)}
    requests = []
    sources = []
    for dt in target_dates:
        d = dt.date()
        if d in schedule:
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
    for row, src in zip(prediction, sources):
        row["schedule_source"] = src
    return {
        "store": store, "start_date": start_date.isoformat(), "days": days,
        "total_predicted_sales": round(float(sum(x["predicted_sales"] for x in prediction)), 2),
        "known_schedule_days": sources.count("test.csv"),
        "assumed_schedule_days": sources.count("assumption"),
        "warning": "Kết quả là ước tính. Ngày không có trong test.csv dùng giả định: nghỉ Chủ nhật, Promo/SchoolHoliday/StateHoliday theo thông số yêu cầu. Dữ liệu lịch sử 2013–2015, không phải dự báo thời gian thực.",
        "points": prediction,
    }


@router.get("/submission")
def download_submission():
    if not SUBMISSION_FILE.exists():
        raise HTTPException(status_code=404, detail="Run python scripts/submission/generate_submission.py first")
    return FileResponse(SUBMISSION_FILE, filename="rossmann_submission.csv", media_type="text/csv")
