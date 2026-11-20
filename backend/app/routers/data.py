from __future__ import annotations
import json
import pandas as pd
from fastapi import APIRouter
from app.services.data_service import (
    METRICS_FILE,
    data_profile,
    get_stores,
    get_test_store_ranges,
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


@router.get("/test-stores")
def test_stores():
    ranges = get_test_store_ranges()
    frame = ranges.merge(
        get_stores()[["Store", "StoreType", "Assortment", "CompetitionDistance"]],
        on="Store",
        how="left",
        validate="one_to_one",
    ).sort_values("Store")
    return [
        {
            "store": int(row.Store),
            "store_type": str(row.StoreType),
            "assortment": str(row.Assortment),
            "competition_distance": (
                None if pd.isna(row.CompetitionDistance) else float(row.CompetitionDistance)
            ),
            "start_date": row.start_date.date().isoformat(),
            "end_date": row.end_date.date().isoformat(),
        }
        for row in frame.itertuples(index=False)
    ]



@router.get("/data-profile")
def profile():
    return data_profile()


@router.get("/model-info")
def model_info():
    if not METRICS_FILE.exists():
        return {"trained": False, "message": "Run python scripts/evaluation/run_evaluation.py"}
    return {"trained": True, **json.loads(METRICS_FILE.read_text(encoding="utf-8"))}
