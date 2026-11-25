FROM python:3.12-slim
WORKDIR /workspace
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir --only-binary=:all: -r backend/requirements.txt
COPY backend/ backend/
COPY scripts/ scripts/
COPY data/raw/ data/raw/
COPY data/external/ data/external/
RUN python scripts/preprocessing/prepare_data.py
COPY models/ models/
COPY results/ results/
WORKDIR /workspace/backend
EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
