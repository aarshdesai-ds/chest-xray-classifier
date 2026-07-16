# Chest X-Ray Classifier — Streamlit deployment image.
# Build from the repo root:  docker build -t chestxray-app .
# Run:                       docker run -d -p 8501:8501 chestxray-app
# Open:                      http://localhost:8501   (on EC2: http://<PUBLIC-IP>:8501)

FROM python:3.11-slim

WORKDIR /app

# System deps: libGL/glib for opencv (Grad-CAM), curl for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching.
COPY phase-7/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + artifacts (kept under /app so app.py's relative paths resolve).
COPY phase-7/app.py .
COPY phase-7/thresholds.json .
COPY phase-7/samples/ ./samples/
COPY models/densenet_finetune_weighted.pt ./models/densenet_finetune_weighted.pt

ENV MODEL_PATH=/app/models/densenet_finetune_weighted.pt \
    THRESH_PATH=/app/thresholds.json

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
