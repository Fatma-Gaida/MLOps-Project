# src/services/tfidf_service.py

import os
import joblib

class TFIDFService:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "tfidf_svm.pkl")
        self.model = joblib.load(model_path)

    def predict(self, text: str):
        """Predict one text."""
        pred = self.model.predict([text])[0]
        probs = self.model.predict_proba([text])[0]
        label_prob = dict(zip(self.model.classes_, probs))
        return {
            "prediction": pred, 
            "probabilities": label_prob
            }

    def predict_batch(self, texts: list):
        """Predict multiple texts."""
        preds = self.model.predict(texts).tolist()
        probs_arr = self.model.predict_proba(texts)
        probabilities = [dict(zip(self.model.classes_, row)) for row in probs_arr]
        return {
            "predictions": preds,
            "probabilities": probabilities
            }
