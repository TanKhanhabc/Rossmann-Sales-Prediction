from datetime import date

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app
from app.routers import predictions
from app.services.data_service import MODEL_FILE

client = TestClient(app)


def test_health_overview_and_stores():
    assert client.get("/api/health").json()["status"] == "ok"
    x = client.get("/api/overview").json()
    assert x["train_rows"] == 1017209
    assert x["test_rows"] == 41088
    assert len(client.get("/api/stores").json()) == 1115


def test_history_and_invalid_store():
    assert len(client.get("/api/history?store=1&days=5").json()["points"]) == 5
    assert client.get("/api/history?store=99999").status_code == 404


def test_predict_and_closed():
    if not MODEL_FILE.exists():
        return
    resp = client.post("/api/predict", json={"store": 1, "date": "2015-08-03", "promo": 1})
    assert resp.status_code == 200, resp.text
    assert resp.json()["predicted_sales"] > 0
    closed = client.post("/api/predict", json={"store": 1, "date": "2015-08-02", "open": 0})
    assert closed.json()["predicted_sales"] == 0
    forecast = client.get("/api/forecast?store=1&start_date=2015-08-01&days=7")
    assert forecast.status_code == 200 and len(forecast.json()["points"]) == 7
    assert forecast.json()["known_schedule_days"] == 7


def test_forecast_uses_test_schedule_for_matching_store_dates(monkeypatch):
    schedule = pd.DataFrame([
        {"Store": 1, "Date": pd.Timestamp("2015-08-01"), "Open": 1, "Promo": 0,
         "StateHoliday": "0", "SchoolHoliday": 1},
        {"Store": 1, "Date": pd.Timestamp("2015-08-03"), "Open": 0, "Promo": 1,
         "StateHoliday": "a", "SchoolHoliday": 0},
    ])
    model_input = {}

    def fake_infer(frame):
        model_input["frame"] = frame
        return np.ones(len(frame))

    monkeypatch.setattr(predictions, "get_test_schedule", lambda: schedule)
    monkeypatch.setattr(predictions, "infer", fake_infer)

    result = predictions.forecast(
        store=1, start_date=date(2015, 8, 1), days=4,
        assumed_promo=1, assumed_school_holiday=1, assumed_state_holiday="b",
    )

    assert result["known_schedule_days"] == 2
    assert result["assumed_schedule_days"] == 2
    assert [point["schedule_source"] for point in result["points"]] == [
        "test.csv", "assumption", "test.csv", "assumption",
    ]
    frame = model_input["frame"]
    assert frame[["Promo", "Open", "StateHoliday", "SchoolHoliday"]].to_dict("records") == [
        {"Promo": 0, "Open": 1, "StateHoliday": "0", "SchoolHoliday": 1},
        {"Promo": 1, "Open": 0, "StateHoliday": "b", "SchoolHoliday": 1},
        {"Promo": 1, "Open": 0, "StateHoliday": "a", "SchoolHoliday": 0},
        {"Promo": 1, "Open": 1, "StateHoliday": "b", "SchoolHoliday": 1},
    ]
