import mlflow
import mlflow.pytorch
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
from datetime import datetime

# Configuration
model_path = "./transformer_model_final"
metrics_path = os.path.join(model_path, "metrics.json")

print("Loading model and metrics...")

# Charger les métriques
with open(metrics_path, "r") as f:
    metrics = json.load(f)

# Charger le modèle et tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Model and tokenizer loaded successfully")

# Configurer MLflow avec chemin absolu
mlflow_dir = os.path.abspath("./mlruns")
mlflow.set_tracking_uri(f"file:///{mlflow_dir}")
print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

# Créer/récupérer l'expérience
experiment_name = "CallCenterAI_Transformer"
experiment = mlflow.set_experiment(experiment_name)
print(f"Experiment: {experiment_name} (ID: {experiment.experiment_id})")

# Démarrer un run
run_name = f"transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
print(f"Starting run: {run_name}")

with mlflow.start_run(run_name=run_name) as run:
    print(f"Run ID: {run.info.run_id}")
    print(f"Artifact URI: {run.info.artifact_uri}")
    
    # Log des hyperparamètres
    print("Logging parameters...")
    mlflow.log_param("model_name", metrics["model_name"])
    mlflow.log_param("num_labels", metrics["num_labels"])
    mlflow.log_param("epochs", 3)
    mlflow.log_param("learning_rate", 2e-5)
    mlflow.log_param("batch_size", 16)
    mlflow.log_param("max_length", 128)
    
    # Log des métriques
    print("Logging metrics...")
    mlflow.log_metric("test_accuracy", metrics["test_accuracy"])
    mlflow.log_metric("test_f1_macro", metrics["test_f1_macro"])
    mlflow.log_metric("test_f1_weighted", metrics["test_f1_weighted"])
    mlflow.log_metric("test_loss", metrics["test_loss"])
    
    # Log du modèle avec signature
    print("Logging model...")
    import pandas as pd
    from mlflow.models.signature import infer_signature
    
    # Exemple d'input/output pour la signature
    sample_input = pd.DataFrame({"text": ["Sample ticket about billing issue"]})
    sample_output = pd.DataFrame({"prediction": [0], "confidence": [0.95]})
    signature = infer_signature(sample_input, sample_output)
    
    # Sauvegarder le modèle
    mlflow.pytorch.log_model(
        model, 
        "model",
        signature=signature,
        registered_model_name="CallCenterAI_Transformer"
    )
    
    # Log des artefacts supplémentaires
    print("Logging artifacts...")
    mlflow.log_artifact(metrics_path, "metrics")
    
    # Log du tokenizer aussi
    tokenizer_dir = os.path.join(model_path, "tokenizer_files")
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)
    mlflow.log_artifacts(tokenizer_dir, "tokenizer")
    
    print("\n" + "="*60)
    print("SUCCESS! Model registered in MLflow")
    print("="*60)
    print(f"MLflow directory: {mlflow_dir}")
    print(f"Experiment: {experiment_name}")
    print(f"Run ID: {run.info.run_id}")
    print(f"Run MLflow UI with:")
    print(f"   mlflow ui --backend-store-uri {mlflow_dir}")
    print(f"   Then open: http://localhost:5000")
    print("="*60)