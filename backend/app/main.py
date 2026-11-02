from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import data, predictions

app = FastAPI(title="Rossmann Sales Forecast API", version="1.0.0",
              description="Forecast daily sales per Rossmann store using store.csv, train.csv and test.csv.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8080"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(data.router)
app.include_router(predictions.router)
