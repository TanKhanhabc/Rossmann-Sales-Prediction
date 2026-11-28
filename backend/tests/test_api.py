from datetime import date

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app
from app.routers import predictions
from app.routers import data
from app.services import data_service
from app.services.data_service import MODEL_FILE

client = TestClient(app)


def test_health_overview_and_stores():
    assert client.get("/api/health").json()["status"] == "ok"
    x = client.get("/api/overview").json()
    assert x["train_rows"] == 1017209
    assert x["test_rows"] == 41088
    assert x["total_sales"] == 5873180623
    assert x["mean_sales_open"] == 6955.51
    assert x["open_days_records"] == 844392
    assert x["daily_average_revenue"] == 6234798.96
    assert x["top_stores"][0] == {"store": 262, "sales": 19516842}
    assert len(client.get("/api/stores").json()) == 1115


def test_test_stores_returns_only_test_stores_with_per_store_date_ranges(monkeypatch):
    schedule = pd.DataFrame([
        {"Store": 2, "Date": pd.Timestamp("2015-08-03")},
        {"Store": 1, "Date": pd.Timestamp("2015-08-01")},
        {"Store": 2, "Date": pd.Timestamp("2015-08-05")},
    ])
    stores = pd.DataFrame([
        {"Store": 1, "StoreType": "a", "Assortment": "a", "CompetitionDistance": 100},
        {"Store": 2, "StoreType": "b", "Assortment": "b", "CompetitionDistance": 200},
        {"Store": 3, "StoreType": "c", "Assortment": "c", "CompetitionDistance": 300},
    ])
    monkeypatch.setattr(
        data,
        "get_test_store_ranges",
        lambda: schedule.groupby("Store", as_index=False)["Date"].agg(
            start_date="min", end_date="max"
        ),
    )
    monkeypatch.setattr(data, "get_stores", lambda: stores)

    response = client.get("/api/test-stores")

    assert response.status_code == 200
    assert response.json() == [
        {"store": 1, "store_type": "a", "assortment": "a", "competition_distance": 100.0,
         "start_date": "2015-08-01", "end_date": "2015-08-01"},
        {"store": 2, "store_type": "b", "assortment": "b", "competition_distance": 200.0,
         "start_date": "2015-08-03", "end_date": "2015-08-05"},
    ]


def test_experiment_comparison_reads_latest_4x5_results():
    response = client.get("/api/experiment-comparison")

    assert response.status_code == 200
    result = response.json()
    assert result["source"] == "results/rossmann_4_experiments_metrics.csv"
    assert len(result["experiments"]) == 4
    assert len(result["models"]) == 5
    assert len(result["rows"]) == 20
    lightgbm_final = next(
        row for row in result["rows"]
        if row["model"] == "LightGBM"
        and row["experiment"] == "TN4: Post-processing (RMSPE)"
    )
    assert lightgbm_final["rmspe"] == 0.11311


def test_unused_model_comparison_endpoint_is_removed():
    assert client.get("/api/model-comparison").status_code == 404


def test_history_endpoint_is_removed():
    assert client.get("/api/history?store=1&days=5").status_code == 404


def test_predict_batch_endpoint_is_removed():
    assert client.post("/api/predict-batch", json=[]).status_code == 404

def test_predict_historical_row_overrides_conditions_and_returns_matching_actual(monkeypatch):
    train = pd.DataFrame([
        {
            "Store": 1,
            "Date": pd.Timestamp("2015-07-31"),
            "Sales": 4321,
            "Promo": 0,
            "Open": 1,
            "SchoolHoliday": 1,
            "StateHoliday": "0",
            "Mean_TemperatureC": 20.0,
        }
    ])
    model_input = {}

    def fake_infer(frame):
        model_input["frame"] = frame
        return np.array([100.0])

    monkeypatch.setattr(predictions, "get_train_for_store", lambda store: train)
    monkeypatch.setattr(predictions, "get_stores", lambda: pd.DataFrame({"Store": [1]}))
    monkeypatch.setattr(predictions, "infer", fake_infer)

    actual = client.post("/api/predict", json={
        "store": 1,
        "date": "2015-07-31",
        "promo": 0,
        "open": 1,
        "school_holiday": 1,
        "state_holiday": "0",
    })
    changed = client.post("/api/predict", json={
        "store": 1,
        "date": "2015-07-31",
        "promo": 1,
        "open": 1,
        "school_holiday": 0,
        "state_holiday": "a",
    })

    assert actual.status_code == 200
    assert actual.json()["actual_sales"] == 4321
    assert actual.json()["conditions_match_actual"] is True
    assert changed.status_code == 200
    assert changed.json()["actual_sales"] is None
    assert changed.json()["conditions_match_actual"] is False
    assert model_input["frame"].iloc[-1][
        ["Promo", "Open", "SchoolHoliday", "StateHoliday"]
    ].to_dict() == {
        "Promo": 1,
        "Open": 1,
        "SchoolHoliday": 0,
        "StateHoliday": "a",
    }


def test_predict_test_row_overrides_conditions_and_never_returns_actual(monkeypatch):
    test = pd.DataFrame([
        {
            "Store": 1,
            "Date": pd.Timestamp("2015-08-03"),
            "Promo": 0,
            "Open": 1,
            "SchoolHoliday": 1,
            "StateHoliday": "0",
            "Mean_TemperatureC": 22.0,
        }
    ])
    model_input = {}

    def fake_infer(frame):
        model_input["frame"] = frame
        return np.array([250.0])

    monkeypatch.setattr(predictions, "get_train_for_store", lambda store: pd.DataFrame())
    monkeypatch.setattr(predictions, "get_test_for_store", lambda store: test)
    monkeypatch.setattr(predictions, "get_stores", lambda: pd.DataFrame({"Store": [1]}))
    monkeypatch.setattr(predictions, "infer", fake_infer)

    response = client.post("/api/predict", json={
        "store": 1,
        "date": "2015-08-03",
        "promo": 1,
        "open": 1,
        "school_holiday": 0,
        "state_holiday": "a",
    })

    assert response.status_code == 200
    assert response.json()["actual_sales"] is None
    assert response.json()["conditions_match_actual"] is False
    assert model_input["frame"].iloc[0][
        ["Promo", "Open", "SchoolHoliday", "StateHoliday"]
    ].to_dict() == {
        "Promo": 1,
        "Open": 1,
        "SchoolHoliday": 0,
        "StateHoliday": "a",
    }


def test_predict_out_of_range_marks_assumption(monkeypatch):
    model_input = {}

    def fake_infer(frame):
        model_input["frame"] = frame
        return np.array([300.0])

    monkeypatch.setattr(predictions, "get_stores", lambda: pd.DataFrame({"Store": [1]}))
    monkeypatch.setattr(predictions, "infer", fake_infer)

    response = client.post("/api/predict", json={
        "store": 1,
        "date": "2026-01-15",
        "promo": 1,
        "open": 1,
        "school_holiday": 0,
        "state_holiday": "0",
    })

    assert response.status_code == 200
    assert response.json()["is_out_of_range"] is True
    assert response.json()["actual_sales"] is None
    assert model_input["frame"].iloc[0]["Date"] == pd.Timestamp("2026-01-15")


def test_submission_endpoint_is_removed():
    assert client.get("/api/submission").status_code == 404


def test_actual_sales_endpoint_is_removed():
    assert client.get(
        "/api/actual-sales?store=1&start_date=2015-07-31&end_date=2015-08-01"
    ).status_code == 404


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
    test_predictions = client.get(
        "/api/test-predictions?store=1&start_date=2015-08-01&end_date=2015-08-01"
    )
    assert test_predictions.status_code == 200, test_predictions.text
    assert len(test_predictions.json()["points"]) == 1
    assert test_predictions.json()["points"][0]["date"] == "2015-08-01"
    assert np.isfinite(test_predictions.json()["points"][0]["predicted_sales"])


def test_test_predictions_uses_test_rows_for_selected_store_and_dates(monkeypatch):
    schedule = pd.DataFrame([
        {"Id": 1, "Store": 1, "DayOfWeek": 6, "Date": pd.Timestamp("2015-08-01"),
         "Open": 1, "Promo": 0, "StateHoliday": "0", "SchoolHoliday": 1},
        {"Id": 2, "Store": 1, "DayOfWeek": 7, "Date": pd.Timestamp("2015-08-02"),
         "Open": 0, "Promo": 0, "StateHoliday": "0", "SchoolHoliday": 1},
        {"Id": 3, "Store": 1, "DayOfWeek": 1, "Date": pd.Timestamp("2015-08-03"),
         "Open": 1, "Promo": 1, "StateHoliday": "0", "SchoolHoliday": 0},
        {"Id": 4, "Store": 2, "DayOfWeek": 6, "Date": pd.Timestamp("2015-08-01"),
         "Open": 1, "Promo": 1, "StateHoliday": "0", "SchoolHoliday": 0},
    ])
    model_input = {}

    def fake_infer(frame):
        model_input["frame"] = frame
        return np.array([120.5, 0])

    monkeypatch.setattr(predictions, "get_stores", lambda: pd.DataFrame({"Store": [1, 2]}))
    monkeypatch.setattr(predictions, "get_test_schedule", lambda: schedule)
    monkeypatch.setattr(
        predictions,
        "get_test_store_ranges",
        lambda: schedule.groupby("Store", as_index=False)["Date"].agg(
            start_date="min", end_date="max"
        ),
    )
    monkeypatch.setattr(predictions, "infer", fake_infer)

    response = client.get(
        "/api/test-predictions?store=1&start_date=2015-08-02&end_date=2015-08-03"
    )

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["total_predicted_sales"] == 120.5
    assert result["points"] == [
        {"date": "2015-08-02", "predicted_sales": 120.5, "open": 0, "promo": 0,
         "state_holiday": "0", "school_holiday": 1},
        {"date": "2015-08-03", "predicted_sales": 0, "open": 1, "promo": 1,
         "state_holiday": "0", "school_holiday": 0},
    ]
    assert model_input["frame"]["Id"].tolist() == [2, 3]
    assert model_input["frame"]["DayOfWeek"].tolist() == [7, 1]


def test_test_predictions_rejects_reversed_or_out_of_test_dates(monkeypatch):
    schedule = pd.DataFrame([
        {"Store": 1, "Date": pd.Timestamp("2015-08-01")},
        {"Store": 1, "Date": pd.Timestamp("2015-08-03")},
    ])
    monkeypatch.setattr(predictions, "get_stores", lambda: pd.DataFrame({"Store": [1]}))
    monkeypatch.setattr(predictions, "get_test_schedule", lambda: schedule)
    monkeypatch.setattr(
        predictions,
        "get_test_store_ranges",
        lambda: schedule.groupby("Store", as_index=False)["Date"].agg(
            start_date="min", end_date="max"
        ),
    )

    reversed_range = client.get(
        "/api/test-predictions?store=1&start_date=2015-08-03&end_date=2015-08-01"
    )
    outside_range = client.get(
        "/api/test-predictions?store=1&start_date=2015-07-31&end_date=2015-08-01"
    )

    assert reversed_range.status_code == 422
    assert outside_range.status_code == 422


def test_test_predictions_validates_dates_against_selected_store(monkeypatch):
    schedule = pd.DataFrame([
        {"Store": 1, "Date": pd.Timestamp("2015-08-01"), "Open": 1, "Promo": 0,
         "StateHoliday": "0", "SchoolHoliday": 0},
        {"Store": 1, "Date": pd.Timestamp("2015-08-03"), "Open": 1, "Promo": 0,
         "StateHoliday": "0", "SchoolHoliday": 0},
        {"Store": 2, "Date": pd.Timestamp("2015-08-05"), "Open": 1, "Promo": 0,
         "StateHoliday": "0", "SchoolHoliday": 0},
        {"Store": 2, "Date": pd.Timestamp("2015-08-07"), "Open": 1, "Promo": 0,
         "StateHoliday": "0", "SchoolHoliday": 0},
    ])
    monkeypatch.setattr(predictions, "get_stores", lambda: pd.DataFrame({"Store": [1, 2]}))
    monkeypatch.setattr(predictions, "get_test_schedule", lambda: schedule)
    monkeypatch.setattr(
        predictions,
        "get_test_store_ranges",
        lambda: schedule.groupby("Store", as_index=False)["Date"].agg(
            start_date="min", end_date="max"
        ),
    )
    monkeypatch.setattr(predictions, "infer", lambda frame: np.zeros(len(frame)))

    before_store_range = client.get(
        "/api/test-predictions?store=2&start_date=2015-08-01&end_date=2015-08-05"
    )
    valid_store_range = client.get(
        "/api/test-predictions?store=2&start_date=2015-08-05&end_date=2015-08-07"
    )

    assert before_store_range.status_code == 422
    assert valid_store_range.status_code == 200
    assert [point["date"] for point in valid_store_range.json()["points"]] == [
        "2015-08-05", "2015-08-07",
    ]


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


def test_forecast_post_prefers_user_schedule_and_defaults_unconfigured_days(monkeypatch):
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

    response = client.post("/api/forecast", json={
        "store": 1,
        "start_date": "2015-08-01",
        "days": 4,
        "schedule": [
            {"date": "2015-08-01", "promo": 1, "open": 0,
             "school_holiday": 0, "state_holiday": "b"},
            {"date": "2015-08-04", "promo": 1, "open": 1,
             "school_holiday": 1, "state_holiday": "c"},
        ],
    })

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["total_predicted_sales"] == 1
    assert result["user_schedule_days"] == 2
    assert result["known_schedule_days"] == 3
    assert result["assumed_schedule_days"] == 1
    assert [point["schedule_source"] for point in result["points"]] == [
        "user", "assumption", "test.csv", "user",
    ]
    assert model_input["frame"][["Promo", "Open", "StateHoliday", "SchoolHoliday"]].to_dict("records") == [
        {"Promo": 1, "Open": 0, "StateHoliday": "b", "SchoolHoliday": 0},
        {"Promo": 0, "Open": 0, "StateHoliday": "0", "SchoolHoliday": 0},
        {"Promo": 1, "Open": 0, "StateHoliday": "a", "SchoolHoliday": 0},
        {"Promo": 1, "Open": 1, "StateHoliday": "c", "SchoolHoliday": 1},
    ]


def test_forecast_post_rejects_duplicate_user_schedule_dates(monkeypatch):
    monkeypatch.setattr(predictions, "get_test_schedule", lambda: pd.DataFrame(columns=[
        "Store", "Date", "Open", "Promo", "StateHoliday", "SchoolHoliday",
    ]))
    payload = {
        "store": 1,
        "start_date": "2015-08-01",
        "days": 2,
        "schedule": [
            {"date": "2015-08-01", "promo": 0, "open": 1,
             "school_holiday": 0, "state_holiday": "0"},
            {"date": "2015-08-01", "promo": 1, "open": 1,
             "school_holiday": 0, "state_holiday": "0"},
        ],
    }

    response = client.post("/api/forecast", json=payload)

    assert response.status_code == 422
