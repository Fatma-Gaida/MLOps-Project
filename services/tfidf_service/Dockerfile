FROM python:3.13.1-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY ./services/tfidf_service/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code only
COPY ./services/tfidf_service/app.py .
COPY ./services/tfidf_service/model_train.py . 
COPY ./services/tfidf_service/classification_report.json .

# Create mlruns directory (but don't require it to exist)
RUN mkdir -p /app/mlruns
COPY ./services/tfidf_service/mlruns /app/mlruns/

# REMOVED: COPY ./services/tfidf_service/models /app/models/

# Verification
RUN echo "=== TF-IDF Service Verification ===" && \
    ls -la /app/ && \
    echo "=== Python files ===" && \
    find /app -name "*.py" -o -name "*.json" | head -20

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]