# Phân chia module 

- Data/EDA: `backend/app/services/data_service.py`, `scripts/data/prepare_data.py`, `scripts/analysis/run_eda.py`, `data/raw/`, `docs/DATA_DICTIONARY.md`.
- ML/Evaluation: `backend/app/ml/`, `scripts/analysis/run_evaluation.py`, `results/`, `models/`.
- API: `backend/app/routers/`, `backend/app/main.py`, `backend/tests/`.
- UI: `frontend/src/` (React TypeScript, Recharts).
- DevOps: Dockerfiles, docker-compose, README.

Có thể giao cho mỗi thành viên 1 module, giữ API contract ổn định.
