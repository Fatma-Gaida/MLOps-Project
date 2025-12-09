# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/transformer

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY ./services/transformer_service/requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create transformer directory and copy if it exists
# NOTE: transformer is in services/transformer_service/transformer/
RUN mkdir -p /app/transformer
COPY ./services/transformer_service/transformer /app/transformer/ 

# Copy service code
COPY ./services/transformer_service/main.py .
COPY ./services/transformer_service/config.py .
COPY ./services/transformer_service/test_api.py .
COPY ./services/transformer_service/mlflow_register.py .

# Create mlruns directory and copy if it exists
RUN mkdir -p /app/mlruns
COPY ./services/transformer_service/mlruns /app/mlruns/ 

# Verification
RUN echo "=== Directory Verification ===" && \
    ls -la /app/ && \
    echo "=== Transformer Contents ===" && \
    ls -la /app/transformer/ 2>/dev/null || echo "Transformer directory empty" && \
    echo "=== MLruns Contents ===" && \
    ls -la /app/mlruns/ 2>/dev/null || echo "MLruns directory empty"

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]