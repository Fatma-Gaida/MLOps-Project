from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time
import os
import json
from contextlib import asynccontextmanager
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import Response
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUEST_COUNT = Counter('transformer_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('transformer_request_latency_seconds', 'Request latency')
MODEL_INFERENCE_TIME = Histogram('transformer_inference_seconds', 'Model inference time')
ACTIVE_REQUESTS = Gauge('transformer_active_requests', 'Number of active requests')
PREDICTION_CONFIDENCE = Histogram('transformer_prediction_confidence', 'Prediction confidence scores')

# ==========================================================
# Variables globales pour le modèle
# ==========================================================
model = None
tokenizer = None
label_mapping = None
MODEL_PATH = os.getenv("MODEL_PATH", "../../transformer_model_final")

# ==========================================================
# Modèles Pydantic
# ==========================================================
class PredictionRequest(BaseModel):
    text: str = Field(..., description="Texte du ticket à classifier", min_length=1)
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Je ne peux pas accéder à mon compte utilisateur"
            }
        }
    }

class PredictionResponse(BaseModel):
    predicted_label: str
    predicted_class: int
    confidence: float
    all_scores: Dict[str, float]
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str

# ==========================================================
# Chargement du modèle
# ==========================================================
def load_model():
    """Charge le modèle Transformer et le tokenizer"""
    global model, tokenizer, label_mapping
    
    try:
        # Convertir en chemin absolu
        model_path = os.path.abspath(MODEL_PATH)
        logger.info(f"Loading model from {model_path}...")
        
        # Vérifier que le chemin existe
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        # Charger le modèle et le tokenizer localement
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        
        # ✅ Mapping personnalisé des labels
        label_mapping = {
            0: "Hardware",
            1: "HR Support",
            2: "Access",
            3: "Miscellaneous",
            4: "Storage",
            5: "Purchase",
            6: "Network",
            7: "Software"
        }

        # Mode évaluation
        model.eval()
        
        # GPU si disponible
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        logger.info(f"✅ Model loaded successfully on {device}")
        logger.info(f"✅ Number of labels: {model.config.num_labels}")
        logger.info(f"✅ Label mapping: {label_mapping}")
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise

# ==========================================================
# Lifecycle - startup / shutdown
# ==========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application"""
    # Startup
    logger.info("🚀 Starting up application...")
    load_model()
    yield
    # Shutdown
    logger.info("👋 Shutting down application...")

# ==========================================================
# Initialisation de l'application FastAPI
# ==========================================================
app = FastAPI(
    title="Transformer Classification Service",
    description="Service de classification de tickets avec modèle Transformer",
    version="1.0.0",
    lifespan=lifespan
)

# ==========================================================
# Configuration CORS
# ==========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# Endpoints
# ==========================================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Vérification de santé du service"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_name": "distilbert-base-multilingual-cased"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Prédiction de la catégorie d'un ticket"""
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    
    try:
        # Tokenization
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        
        # Déplacer vers le bon device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Inférence
        inference_start = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
        
        inference_time = time.time() - inference_start
        MODEL_INFERENCE_TIME.observe(inference_time)
        
        # Récupérer la prédiction
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        # Créer le dictionnaire de tous les scores
        all_scores = {
            label_mapping.get(i, f"Class_{i}"): probabilities[0][i].item()
            for i in range(len(probabilities[0]))
        }
        
        predicted_label = label_mapping.get(predicted_class, f"Unknown_{predicted_class}")
        
        # Métriques
        processing_time = time.time() - start_time
        REQUEST_LATENCY.observe(processing_time)
        PREDICTION_CONFIDENCE.observe(confidence)
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="success").inc()
        
        logger.info(f"✅ Prediction: {predicted_label} (confidence: {confidence:.4f})")
        
        return {
            "predicted_label": predicted_label,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "all_scores": {k: round(v, 4) for k, v in all_scores.items()},
            "processing_time": round(processing_time, 4)
        }
        
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="error").inc()
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        ACTIVE_REQUESTS.dec()

@app.get("/metrics")
async def metrics():
    """Endpoint pour Prometheus"""
    return Response(content=generate_latest(), media_type="text/plain")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Transformer Classification Service",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }

# ==========================================================
# Lancement de l'application
# ==========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
