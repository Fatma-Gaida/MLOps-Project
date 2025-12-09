FROM python:3.13.1-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from service directory
COPY services/tfidf_service/requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code from service directory
COPY services/tfidf_service/app.py .

# Create mlruns directory and copy if it exists
RUN mkdir -p /app/mlruns
COPY services/tfidf_service/mlruns /app/mlruns/

# Verification - show mlruns structure
RUN echo "=== MLflow Directory Verification ===" && \
    ls -la /app/ && \
    echo "=== MLruns Contents ===" && \
    ls -la /app/mlruns/ 2>/dev/null || echo "No mlruns found"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]