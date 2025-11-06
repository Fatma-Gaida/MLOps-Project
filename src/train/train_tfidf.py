import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import mlflow


df = pd.read_csv("../../data/tickets.csv")
X = df["Document"]
y = df["Topic_group"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1,2))),
    ("svm", CalibratedClassifierCV(LinearSVC(), cv=3))
])



with mlflow.start_run():
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Log model
    mlflow.sklearn.log_model(pipeline, "tfidf_svm_model")

    # ===== Save local model for API =====
    os.makedirs("../models", exist_ok=True)
    joblib.dump(pipeline, "../models/tfidf_svm.pkl")

print("Accuracy:", acc, "F1:", f1)