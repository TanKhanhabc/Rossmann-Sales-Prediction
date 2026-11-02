"""Inference helpers shared by API and scripts."""
from app.ml.model_service import infer, load_model, predict_frame
__all__ = ["infer", "load_model", "predict_frame"]
