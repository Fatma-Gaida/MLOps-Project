FROM python:3.13.1-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Copy mlruns from the parent directory (project root)
# The context should be set to the project root when building
COPY mlruns /app/mlruns

# Verification - show mlruns structure
RUN echo "=== MLflow Directory Verification ===" && \
    ls -la /app/ && \
    echo "=== MLruns Contents ===" && \
    ls -R /app/mlruns | head -50

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]