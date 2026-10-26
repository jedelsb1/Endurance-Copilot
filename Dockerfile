FROM python:3.11-slim

WORKDIR /app

# Install sqlite3, tzdata and system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/knowledge /app/data/strava_cache /app/data/fit_cache/processed /app/data/fit_dropzone/processed /root/.garminconnect && \
    chmod -R 777 /app

ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV TZ="America/Los_Angeles"
EXPOSE 7860

CMD ["sh", "-c", "python scripts/migrate_json_to_sqlite.py && uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
