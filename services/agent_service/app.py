from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Union
import os
from groq import Groq
import re
import requests
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

app = FastAPI(title="Agent IA Service")

# Environment variables for service URLs
TFIDF_SERVICE_URL = os.getenv("TFIDF_SERVICE_URL", "http://127.0.0.1:8002")
TRANSFORMER_SERVICE_URL = os.getenv("TRANSFORMER_SERVICE_URL", "http://127.0.0.1:8001")

# Groq API key
GROQ_API_KEY = "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"

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Prometheus metrics
REQUEST_COUNTER = Counter('agent_requests_total', 'Total requests to agent service', ['model_chosen'])
REQUEST_LATENCY = Histogram('agent_request_latency_seconds', 'Request latency', ['model_chosen'])

class TicketRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    category: str
    confidence: float
    model_used: str
    explanation: str
    scrubbed_text: str
    model_output: Dict[str, Any]  # Add this field to return full model response

def scrub_pii(text: str) -> str:
    """
    Simple PII scrubbing using regex to mask emails, phone numbers, and potential names/IPs.
    This is a basic implementation; for production, use a library like Presidio.
    """
    # Mask emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    # Mask phone numbers (basic pattern)
    text = re.sub(r'\b(?:\+?(\d{1,3}))?[-. (]*(\d{3})?[-. )]*(\d{3})[-. ]*(\d{4})\b', '[PHONE]', text)
    # Mask IP addresses
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP]', text)
    # Mask potential credit card numbers (basic)
    text = re.sub(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', '[CARD]', text)
    return text

def decide_model(text: str) -> str:
    prompt = f"""
You are an expert intelligent routing system that decides whether to use TFIDF or TRANSFORMER to classify a text.

Rules:
1. TFIDF is used only for short (<=20 words), simple English texts.
2. TRANSFORMER is used for:
   - Texts longer than 20 words
   - Texts containing technical, nuanced, or complex meaning
   - Texts in any language other than English
3. Always respond with exactly one word: TFIDF or TRANSFORMER. Do NOT include any extra explanation.

Examples:
Text: "Forgot password on my laptop"
Answer: TFIDF

Text: "Le système plante lorsque plusieurs utilisateurs téléchargent de gros fichiers simultanément"
Answer: TRANSFORMER

Text: "Cannot access Outlook account from company laptop"
Answer: TFIDF

Text: "The system crashes when multiple users upload large datasets concurrently"
Answer: TRANSFORMER

Now decide for the following text:
Text: "{text}"
Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=30,
        temperature=0.7
    )

    result = response.choices[0].message.content.strip().upper()
    if "TRANSFORMER" in result:
        return "transformer"
    return "tfidf"

def call_model_service(model: str, text: str) -> dict:
    if model == "tfidf":
        url = f"{TFIDF_SERVICE_URL}/predict"
        payload = {"text": text}
    elif model == "transformer":
        url = f"{TRANSFORMER_SERVICE_URL}/predict"
        payload = {"text": text}
    else:
        raise ValueError("Invalid model selected")

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Model service unavailable: {str(e)}")

    data = response.json()
    
    # Handle different response formats from different services
    category = (data.get("predicted_label") or   # Transformer format
                data.get("predicted_class") or   # TF-IDF format
                data.get("label") or 
                data.get("category"))
    
    confidence = (data.get("confidence") or      # Both use 'confidence'
                  data.get("proba") or 
                  data.get("score"))
    
    if category is None or confidence is None:
        raise HTTPException(
            status_code=500, 
            detail=f"Invalid response from {model} service. Got: {data}"
        )
    
    # Return both extracted values and full model output
    return {
        "category": category, 
        "confidence": float(confidence),
        "full_response": data  # Keep the complete model response
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(ticket: TicketRequest):
    scrubbed_text = scrub_pii(ticket.text)
    
    with REQUEST_LATENCY.labels(model_chosen='pending').time():
        model = decide_model(scrubbed_text)
    
    explanation = f"Selected {model.upper()} based on text length, complexity, and language."
    
    with REQUEST_LATENCY.labels(model_chosen=model).time():
        prediction = call_model_service(model, scrubbed_text)
    
    REQUEST_COUNTER.labels(model_chosen=model).inc()
    
    return PredictionResponse(
        category=prediction["category"],
        confidence=prediction["confidence"],
        model_used=model,
        explanation=explanation,
        scrubbed_text=scrubbed_text,
        model_output=prediction["full_response"]  # Include full model response
    )

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)