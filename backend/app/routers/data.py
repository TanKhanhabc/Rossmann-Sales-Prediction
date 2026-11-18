from __future__ import annotations
from datetime import date
import json
from fastapi import APIRouter, HTTPException, Query
from app.services.data_service import (
    METRICS_FILE,
    data_profile,
    get_actual_sales,
    get_stores,
    get_train,
    overview,
)

router = APIRouter(prefix="/api", tags=["Data & Analytics"])


@router.get("/health")
def health():
    return {"status": "ok", "project": "Rossmann Sales Forecast"}


@router.get("/overview")
def get_overview():
    return overview()


@router.get("/stores")
def stores():
    frame = get_stores()[["Store", "StoreType", "Assortment", "CompetitionDistance"]]
    return [{"store": int(r.Store), "store_type": str(r.StoreType),
             "assortment": str(r.Assortment),
             "competition_distance": None if r.CompetitionDistance != r.CompetitionDistance else float(r.CompetitionDistance)}
            for r in frame.itertuples(index=False)]


@router.get("/actual-sales")
def actual_sales(
    store: int = Query(..., ge=1),
    start_date: date | None = None,
    end_date: date | None = None,
):
    if store not in set(get_stores().Store):
        raise HTTPException(status_code=404, detail="Không tìm thấy cửa hàng")

    train = get_train()
    train_start = train.Date.min().date()
    train_end = train.Date.max().date()
    start_date = start_date or train_start
    end_date = end_date or train_end
    if start_date > end_date:
        raise HTTPException(status_code=422, detail="Ngày bắt đầu phải trước hoặc bằng ngày kết thúc")
    if start_date < train_start or end_date > train_end:
        raise HTTPException(
            status_code=422,
            detail=f"Khoảng ngày phải nằm trong dữ liệu train ({train_start} – {train_end})",
        )

    return {
        "store": store,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "points": get_actual_sales(store, start_date, end_date),
    }


@router.get("/data-profile")
def profile():
    return data_profile()


@router.get("/model-info")
def model_info():
    if not METRICS_FILE.exists():
        return {"trained": False, "message": "Run python scripts/evaluation/run_evaluation.py"}
    return {"trained": True, **json.loads(METRICS_FILE.read_text(encoding="utf-8"))}
