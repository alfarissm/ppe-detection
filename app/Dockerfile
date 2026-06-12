# HuggingFace Spaces (Docker) / generic container deploy
FROM python:3.11-slim

# libs needed by opencv / ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces expects 7860; override with PORT elsewhere
ENV PORT=7860
EXPOSE 7860

# gunicorn: 1 worker (model loaded once), long timeout for inference
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 app:app
