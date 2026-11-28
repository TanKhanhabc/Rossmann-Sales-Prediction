from __future__ import annotations
import json
import pandas as pd
from fastapi import APIRouter
from app.services.data_service import (
    METRICS_FILE,
    REPORTS_DIR,
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


@router.get("/experiment-comparison")
def experiment_comparison():
    csv_path = REPORTS_DIR.parent / "results" / "rossmann_4_experiments_metrics.csv"
    if not csv_path.is_file():
        return {
            "experiments": [],
            "models": [],
            "rows": [],
            "message": "Chưa tìm thấy kết quả thực nghiệm mới.",
            "missing_files": [str(csv_path)],
        }

    frame = pd.read_csv(csv_path)
    required = {"Thực nghiệm", "Mô hình", "RMSPE", "Train Time (s)", "Inference (ms)", "Size (MB)"}
    if not required.issubset(frame.columns):
        return {
            "experiments": [],
            "models": [],
            "rows": [],
            "message": "Định dạng file kết quả thực nghiệm không hợp lệ.",
            "missing_files": [],
        }

    experiments = [
        str(value)
        for value in dict.fromkeys(frame["Thực nghiệm"].fillna("").astype(str).tolist())
        if value
    ]
    models = [
        str(value)
        for value in dict.fromkeys(frame["Mô hình"].fillna("").astype(str).tolist())
        if value
    ]
    rows = [
        {
            "experiment": str(record["Thực nghiệm"]),
            "model": str(record["Mô hình"]),
            "rmspe": round(float(record["RMSPE"]), 5),
            "train_time": round(float(record["Train Time (s)"]), 2),
            "inference_ms": round(float(record["Inference (ms)"]), 2),
            "size_mb": round(float(record["Size (MB)"]), 3),
        }
        for record in frame.to_dict("records")
    ]
    return {
        "experiments": experiments,
        "models": models,
        "rows": rows,
        "source": "results/rossmann_4_experiments_metrics.csv",
    }
