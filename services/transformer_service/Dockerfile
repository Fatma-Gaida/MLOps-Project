# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

WORKDIR /app

# Installer uniquement les dépendances de compilation nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les requirements
COPY ./services/transformer_service/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# Deuxième stage - image finale
FROM python:3.11-slim

WORKDIR /app

# Copier les packages Python installés
COPY --from=builder /root/.local /root/.local

# Ajouter au PATH
ENV PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/root/.local/lib/python3.11/site-packages:$PYTHONPATH

# Installer les dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier uniquement les fichiers nécessaires
COPY ./services/transformer_service/main.py .
COPY ./services/transformer_service/config.py .

# Supprimer les fichiers de développement PyTorch inutiles
RUN find /root/.local -name "*.pyc" -delete && \
    find /root/.local -name "__pycache__" -type d -exec rm -rf {} + && \
    find /root/.local -name "tests" -type d -exec rm -rf {} + && \
    find /root/.local -name "*.so" ! -name "*.cpython*.so" -delete

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]