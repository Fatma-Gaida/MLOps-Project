# services/tfidf_service/train.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import mlflow
from mlflow.models import infer_signature
import logging
import os
from pathlib import Path

# ==================== CONFIGURATION & LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Create mlruns folder and set tracking URI (works on Windows + Docker)
MLRUNS_DIR = Path("../../mlruns").resolve()
MLRUNS_DIR.mkdir(exist_ok=True)
tracking_uri = f"file:///{MLRUNS_DIR.as_posix().replace(os.sep, '/')}"
mlflow.set_tracking_uri(tracking_uri)
logger.info(f"MLflow tracking URI set to: {tracking_uri}")

# Set or create experiment
EXPERIMENT_NAME = "Ticket_Classification_TFIDF"
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    logger.info(f"Created new experiment: {EXPERIMENT_NAME}")
else:
    experiment_id = experiment.experiment_id
    logger.info(f"Using existing experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")

mlflow.set_experiment(EXPERIMENT_NAME)

# ==================== DATA LOADING ====================
data_path = "../../data/tickets.csv"
logger.info(f"Loading data from {data_path}...")
df = pd.read_csv(data_path, encoding='utf-8')

logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
logger.info(f"Columns: {list(df.columns)}")
logger.info(f"Missing values in 'Document': {df['Document'].isna().sum()}")
logger.info(f"Missing values in 'Topic_group': {df['Topic_group'].isna().sum()}")

# Clean data
df = df.dropna(subset=['Document', 'Topic_group'])
df['Document'] = df['Document'].astype(str)
logger.info(f"After cleaning: {df.shape[0]} valid samples")
logger.info(f"Classes distribution:\n{df['Topic_group'].value_counts()}")

X = df['Document']
y = df['Topic_group']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
logger.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

# ==================== MODEL TRAINING ====================
logger.info("Building and training TF-IDF + Calibrated LinearSVC model...")

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        lowercase=True,
        strip_accents='unicode',
        sublinear_tf=True
    )),
    ('clf', CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=5000, random_state=42, class_weight='balanced'),
        method='sigmoid',
        cv=3
    ))
])

pipeline.fit(X_train, y_train)
logger.info("Model training completed!")

# ==================== EVALUATION ====================
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline[-1], "predict_proba") else None

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

logger.info(f"Accuracy: {acc:.4f}")
logger.info(f"F1-weighted: {f1:.4f}")
print("\n" + classification_report(y_test, y_pred))

# ==================== INPUT EXAMPLE & SIGNATURE ====================
input_example = ["I cannot login to my account because password expired and I forgot security questions"]
predictions_example = pipeline.predict(input_example)
signature = infer_signature(input_example, predictions_example)

# ==================== MLFLOW LOGGING ====================
model_name = "TFIDF_SVM_Ticket_Classifier"

with mlflow.start_run(run_name="tfidf_svm_final_v1"):
    # Log parameters
    mlflow.log_params({
        "max_features": 15000,
        "ngram_range": "(1,2)",
        "svm_C": 1.0,
        "svm_max_iter": 5000,
        "calibration": "sigmoid",
        "cv_folds": 3,
        "class_weight": "balanced",
        "dataset_rows": len(df),
        "test_size": 0.2
    })

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_weighted", f1)

    # Log classification report as artifact
    report = classification_report(y_test, y_pred, output_dict=True)
    import json
    with open("classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    mlflow.log_artifact("classification_report.json")

    # Log model with proper registry handling
    logger.info(f"Logging model to MLflow Model Registry as: {model_name}")

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
        registered_model_name=model_name,  # Creates or adds new version
        await_registration_for=60
    )

    # Get the latest version
    client = mlflow.tracking.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
    logger.info(f"Model successfully registered: {model_name} v{latest_version}")

    logger.info("Run completed and logged in MLflow!")

# ==================== FINAL SUCCESS MESSAGE ====================
print("\n" + "="*70)
print("                TOUT EST PARFAIT !")
print(f"   Modèle enregistré : {model_name} (version {latest_version})")
print("   Ouvre ton UI → http://localhost:5000")
print("   Lance ton service FastAPI :")
print("       uvicorn app:app --reload --port 8002")
print("   Le endpoint /predict fonctionnera à 100% !")
print("="*70)