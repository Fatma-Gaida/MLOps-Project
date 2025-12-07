import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
import glob
from prometheus_fastapi_instrumentator import Instrumentator
import logging

# ==================== LOGGING CONFIG ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ==================== MLFLOW CONFIG ====================
MODEL_NAME = "TFIDF_SVM_Ticket_Classifier"

# Get current directory and construct mlruns path
current_dir = os.path.dirname(os.path.abspath(__file__))
mlruns_path = os.path.join(current_dir, "mlruns")

logger.info(f"📂 Current directory: {current_dir}")
logger.info(f"📂 MLruns path: {mlruns_path}")
logger.info(f"📂 MLruns exists: {os.path.exists(mlruns_path)}")

# ==================== LOAD MODEL ====================
def find_model_path():
    """
    Find model in mlruns structure - checks both models folder and experiment runs
    """
    
    # Strategy 1: Check registered models folder
    models_registry_path = os.path.join(mlruns_path, "models", MODEL_NAME)
    if os.path.exists(models_registry_path):
        logger.info(f"🔍 Found models registry: {models_registry_path}")
        
        # Find version folders
        version_folders = glob.glob(os.path.join(models_registry_path, "version-*"))
        if version_folders:
            # Use the latest version
            latest_version = sorted(version_folders)[-1]
            logger.info(f"📦 Found version folder: {latest_version}")
            
            # The meta.yaml in version folder points to the actual artifacts
            # We need to find the artifacts/model folder
            # First, check if there's a direct artifacts path
            artifacts_path = os.path.join(latest_version, "artifacts", "model")
            if os.path.exists(artifacts_path):
                logger.info(f"✅ Found model at: {artifacts_path}")
                return artifacts_path
            
            # Sometimes the model is stored elsewhere, read meta.yaml to find source
            meta_file = os.path.join(latest_version, "meta.yaml")
            if os.path.exists(meta_file):
                import yaml
                with open(meta_file, 'r') as f:
                    meta = yaml.safe_load(f)
                    if 'source' in meta:
                        source_uri = meta['source']
                        logger.info(f"📋 Meta source: {source_uri}")
                        
                        # Parse the source URI to get run_id
                        # Format is usually: file:///.../mlruns/EXP_ID/RUN_ID/artifacts/model
                        if 'mlruns' in source_uri:
                            parts = source_uri.split('mlruns')[-1].split('/')
                            parts = [p for p in parts if p]
                            if len(parts) >= 3:
                                exp_id = parts[0]
                                run_id = parts[1]
                                model_path = os.path.join(mlruns_path, exp_id, run_id, "artifacts", "model")
                                if os.path.exists(model_path):
                                    logger.info(f"✅ Found model via meta.yaml: {model_path}")
                                    return model_path
    
    # Strategy 2: Search all experiment runs
    logger.info("🔍 Searching experiment runs...")
    for item in os.listdir(mlruns_path):
        exp_path = os.path.join(mlruns_path, item)
        
        # Skip non-directories and special folders
        if not os.path.isdir(exp_path) or item in ['.trash', 'models', '.mflow.deletions', '0']:
            continue
        
        logger.info(f"🔍 Checking experiment: {item}")
        
        # Look for run folders
        try:
            for run_id in os.listdir(exp_path):
                run_path = os.path.join(exp_path, run_id)
                model_path = os.path.join(run_path, "artifacts", "model")
                
                if os.path.exists(model_path) and os.path.isdir(model_path):
                    # Check if MLmodel file exists (confirms it's a valid model)
                    if os.path.exists(os.path.join(model_path, "MLmodel")):
                        logger.info(f"✅ Found model in run: {run_id}")
                        return model_path
        except Exception as e:
            logger.warning(f"⚠️ Error scanning {exp_path}: {e}")
            continue
    
    raise RuntimeError(f"❌ No model found in: {mlruns_path}")

# Find and load the model
try:
    model_path = find_model_path()
    logger.info(f"📦 Loading model from: {model_path}")
    
    # Load model directly from path
    model = mlflow.sklearn.load_model(model_path)
    
    logger.info(f"✅ Model loaded successfully!")
    logger.info(f"✅ Model classes: {model.classes_}")
    model_version = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
    
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    raise

# ==================== FASTAPI ====================
app = FastAPI(
    title="TFIDF Ticket Classifier",
    version="1.0.0",
    docs_url="/docs",
)

Instrumentator().instrument(app).expose(app)

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: List[dict]

@app.get("/")
def health():
    return {
        "status": "alive", 
        "model": MODEL_NAME,
        "model_path": model_path,
        "classes": model.classes_.tolist()
    }

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    txt = payload.text.strip()
    if not txt:
        raise HTTPException(400, "Empty text")

    pred = model.predict([txt])[0]
    probas = model.predict_proba([txt])[0]
    confidence = float(max(probas))

    return PredictResponse(
        predicted_class=pred,
        confidence=confidence,
        probabilities=[
            {"label": lbl, "probability": float(p)}
            for lbl, p in zip(model.classes_, probas)
        ]
    )