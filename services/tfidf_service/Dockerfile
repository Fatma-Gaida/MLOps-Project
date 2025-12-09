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

# Copy application code
COPY ./services/tfidf_service/app.py .
COPY ./services/tfidf_service/model_train.py .
COPY ./services/tfidf_service/classification_report.json . 

# Copy models if they exist
RUN mkdir -p /app/models
COPY ./services/tfidf_service/models /app/models/ 

# Create mlruns directory
RUN mkdir -p /app/mlruns
COPY ./services/tfidf_service/mlruns /app/mlruns/ 

# Verification
RUN echo "=== TF-IDF Service Verification ===" && \
    ls -la /app/ && \
    ls -la /app/mlruns/ 2>/dev/null || echo "MLruns directory empty" && \
    ls -la /app/models/ 2>/dev/null || echo "Models directory empty"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]