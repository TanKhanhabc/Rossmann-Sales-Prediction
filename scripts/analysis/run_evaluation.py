"""Run from project root: python scripts/analysis/run_evaluation.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.ml.model_service import train_and_evaluate

if __name__ == "__main__":
    train_and_evaluate(validation_days=42)
