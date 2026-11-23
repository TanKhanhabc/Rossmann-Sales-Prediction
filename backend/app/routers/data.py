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


@router.get("/model-comparison")
def model_comparison():
    cv_path = REPORTS_DIR / "cv_results.csv"
    holdout_path = REPORTS_DIR / "final_test_results.csv"
    selected_path = REPORTS_DIR / "selected_model.json"
    required = {
        "cv": cv_path,
        "holdout": holdout_path,
        "selection": selected_path,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        return {"models": [], "best_model": None, "stages": {}, "message": "Chạy các script thử nghiệm mô hình để tạo báo cáo.", "missing_files": missing}

    cv = pd.read_csv(cv_path)
    holdout = pd.read_csv(holdout_path)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    holdout_by_name = {str(row.Model).strip().lower(): row for row in holdout.itertuples(index=False)}
    cv_models = []
    for row in cv.sort_values("Val_MAE").itertuples(index=False):
        name = str(row.Model).strip().lower()
        cv_models.append({
            "name": name,
            "train_mae": round(float(row.Train_MAE), 5),
            "val_mae": round(float(row.Val_MAE), 5),
            "val_mae_sd": round(float(row.Val_MAE_SD), 5),
            "rmse": round(float(row.RMSE), 5),
            "r2": round(float(row.R2), 5),
            "within_1sd": bool(row.Within_1SD),
            "selected": bool(row.Selected),
            "best_params": str(row.Best_Params),
        })

    holdout_models = []
    for row in holdout.itertuples(index=False):
        holdout_models.append({
            "name": str(row.Model).strip().lower(),
            "mae": round(float(row.mae), 5),
            "rmse": round(float(row.rmse), 5),
            "rmspe": round(float(row.rmspe), 5),
            "wape": round(float(row.wape), 5),
            "mae_ci95": [round(float(row.MAE_CI95_Lower), 5), round(float(row.MAE_CI95_Upper), 5)],
            "rmse_ci95": [round(float(row.RMSE_CI95_Lower), 5), round(float(row.RMSE_CI95_Upper), 5)],
        })

    subgroup_tables = {}
    for dimension, filename in (
        ("store_type", "subgroup_mae_storetype.csv"),
        ("day_of_week", "subgroup_mae_dayofweek.csv"),
        ("promo", "subgroup_mae_promo.csv"),
    ):
        path = REPORTS_DIR / filename
        if path.is_file():
            frame = pd.read_csv(path)
            subgroup_tables[dimension] = [
                {str(key): (round(float(value), 5) if isinstance(value, float) else int(value) if str(key).lower() == "rows" else str(value))
                 for key, value in record.items()}
                for record in frame.to_dict("records")
            ]

    interval_summary = []
    interval_path = REPORTS_DIR / "prediction_interval_summary.csv"
    if interval_path.is_file():
        interval_summary = [
            {key: (round(float(value), 5) if isinstance(value, (float, int)) and str(key) != "interval_method" else str(value))
             for key, value in record.items()}
            for record in pd.read_csv(interval_path).to_dict("records")
        ]

    importance = []
    importance_path = REPORTS_DIR / "permutation_importance.csv"
    if importance_path.is_file():
        frame = pd.read_csv(importance_path).head(20)
        importance = [
            {
                "feature": str(row.Feature),
                "importance_mae_increase": round(float(row.Importance_MAE_Increase), 5),
                "importance_sd": round(float(row.Importance_SD), 5),
            }
            for row in frame.itertuples(index=False)
        ]

    metadata = {}
    metadata_path = REPORTS_DIR / "final_test_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    top_errors = []
    top_errors_path = REPORTS_DIR / "top_10_errors.csv"
    if top_errors_path.is_file():
        top_errors = [
            {key: (round(float(value), 5) if isinstance(value, float) else int(value) if str(key).lower() in {"store", "dayofweek", "promo", "open"} else str(value))
             for key, value in record.items()}
            for record in pd.read_csv(top_errors_path).head(10).to_dict("records")
        ]

    selection_summary = []
    selection_summary_path = REPORTS_DIR / "selection_summary.csv"
    if selection_summary_path.is_file():
        selection_frame = pd.read_csv(selection_summary_path).fillna("")
        selection_summary = []
        for record in selection_frame.to_dict("records"):
            normalized = {}
            for key, value in record.items():
                key_lower = str(key).lower()
                if key_lower in {"within_1sd", "selected"}:
                    normalized[key] = str(value).strip().lower() in {"true", "1", "yes"}
                elif isinstance(value, (float, int)) and not isinstance(value, bool):
                    normalized[key] = round(float(value), 5)
                else:
                    normalized[key] = str(value)
            selection_summary.append(normalized)

    best_name = str(selected.get("model", "")).strip().lower() or None
    return {
        "models": cv_models,
        "best_model": best_name,
        "stages": {
            "cross_validation": cv_models,
            "holdout": holdout_models,
            "subgroups": subgroup_tables,
            "feature_importance": importance,
            "top_errors": top_errors,
            "selection": selection_summary,
        },
        "interval_summary": interval_summary,
        "metadata": metadata,
        "selected_reason": str(selected.get("selection_reason", "")),
        "sources": {
            "cv": "reports/cv_results.csv",
            "holdout": "reports/final_test_results.csv",
            "selection": "reports/selected_model.json",
        },
    }


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
