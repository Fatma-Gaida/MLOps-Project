from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time

# Import your updated AgentService that uses Groq
from src.services.agent_service import AgentService

# -------- Init App --------
app = FastAPI(
    title="TF-IDF / Transformer API",
    description="Predict ticket categories using TF-IDF or Transformer based on text complexity",
    version="1.0"
)

# Enable CORS for all origins (can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Services --------
# This will use the new Groq-powered AgentService
agent_service = AgentService()

# -------- Pydantic Schemas --------
class TextRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: List[str]

class GenerateRequest(BaseModel):
    prompt: str

# -------- Prometheus metrics --------
PREDICTIONS_TOTAL = Counter("predictions_total", "Total number of predictions")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Latency of prediction")

# -------- Routes --------
@app.get("/ping")
def ping():
    return {"status": "ok", "message": "API is running"}

@app.post("/analyze")
def analyze(request: TextRequest):
    """
    Main endpoint — Groq decides which model handles the text.
    """
    try:
        start_time = time.time()
        result = agent_service.route_text(request.text)

        PREDICTIONS_TOTAL.inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)

        return result
    except Exception as e:
        print("🔥 ERROR in /analyze:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/metrics")
def metrics():
    """
    Prometheus metrics endpoint
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
