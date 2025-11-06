from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
from typing import List
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time
from fastapi.responses import Response

# -------- Init App --------
app = FastAPI(title="TF-IDF SVM API",description="Predict simple ticket categories", version="1.0")

# Enable CORS for all origins (can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Load Model --------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "tfidf_svm.pkl")
model = joblib.load(MODEL_PATH)

# -------- Pydantic Schemas --------
class TextRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts : List[str]

# -------- Prometheus metrics --------
PREDICTIONS_TOTAL = Counter("tfidf_predictions_total", "Total number of predictions")
PREDICTION_LATENCY = Histogram("tfidf_prediction_latency_seconds", "Latency of prediction")

# -------- Routes --------
@app.get("/ping")
def ping():
    return {"status": "ok", "message": "API is running"}

@app.post("/predict")
def predict(request: TextRequest):
    try:
        start_time = time.time()
        # Make prediction
        preds = model.predict([request.text])[0]  # single label
        probs = model.predict_proba([request.text])[0]  # probability array
        label_prob = dict(zip(model.classes_, probs))  # map labels to probabilities

        # Update Prometheus metrics
        PREDICTIONS_TOTAL.inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)

        # Return prediction + probabilities
        return {
            "prediction": preds,
            "probabilities": label_prob
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict_batch")
def predict_batch(request: BatchRequest):
    try:
        start_time = time.time()
        predictions = model.predict(request.texts).tolist()
        probabilities_array = model.predict_proba((request.texts))

        # Convert each row to a dict of label -> probability
        probabilities = [dict(zip(model.classes_, row)) for row in probabilities_array]

        PREDICTIONS_TOTAL.inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)

        return {
            "predictions": predictions,
            "probabilities": probabilities
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
