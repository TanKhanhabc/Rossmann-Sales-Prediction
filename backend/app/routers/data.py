from __future__ import annotations
import json
from fastapi import APIRouter, HTTPException, Query
from app.services.data_service import get_stores, overview, get_history, data_profile, METRICS_FILE

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


@router.get("/history")
def history(store: int | None = None, days: int = Query(45, ge=1, le=365)):
    if store is not None and store not in set(get_stores().Store):
        raise HTTPException(status_code=404, detail="Không tìm thấy cửa hàng")
    return {"store": store, "days": days, "points": get_history(store, days)}


@router.get("/data-profile")
def profile():
    return data_profile()


@router.get("/model-info")
def model_info():
    if not METRICS_FILE.exists():
        return {"trained": False, "message": "Run python scripts/analysis/run_evaluation.py"}
    return {"trained": True, **json.loads(METRICS_FILE.read_text(encoding="utf-8"))}
