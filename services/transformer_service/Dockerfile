# services/transformer_service/Dockerfile
FROM python:3.11-slim AS builder

WORKDIR /app

# Étape 1: Installer PyTorch CPU SEULEMENT
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.1.1 --index-url https://download.pytorch.org/whl/cpu

# Étape 2: Installer le reste depuis requirements.txt
COPY ./services/transformer_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Étape 3: Nettoyer les fichiers inutiles de PyTorch
RUN find /usr/local/lib/python3.11 -name "*.pyc" -delete && \
    find /usr/local/lib/python3.11 -name "__pycache__" -type d -exec rm -rf {} + && \
    rm -rf /usr/local/lib/python3.11/site-packages/torch/test* 2>/dev/null || true && \
    rm -rf /usr/local/lib/python3.11/site-packages/torch/*.a 2>/dev/null || true

# Deuxième stage : image finale
FROM python:3.11-slim

WORKDIR /app

# Copier uniquement les packages installés
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copier votre code
COPY ./services/transformer_service/main.py .
COPY ./services/transformer_service/config.py .
COPY ./services/transformer_service/test_api.py . 
COPY ./services/transformer_service/mlflow_register.py . 

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]