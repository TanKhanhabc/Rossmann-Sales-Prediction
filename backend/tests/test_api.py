from fastapi.testclient import TestClient
from app.main import app
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
