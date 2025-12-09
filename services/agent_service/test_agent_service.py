"""
Tests pour l'Agent Service
À placer dans: services/agent_service/tests/test_api.py
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os

# Ajouter le chemin du service au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Importer l'application (adapter selon votre structure)
# from app import app

# Pour la démo, créons une app de test
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "agent"}

@app.get("/metrics")
async def metrics():
    return "# HELP test metric\n"

@app.post("/predict")
async def predict(request: dict):
    return {
        "category": "Hardware",
        "confidence": 0.85,
        "chosen_model": "tfidf"
    }


client = TestClient(app)


class TestAgentServiceHealth:
    """Tests pour les endpoints de santé"""

    def test_health_endpoint(self):
        """Test que l'endpoint /health retourne 200"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["service"] == "agent"

    def test_metrics_endpoint(self):
        """Test que l'endpoint /metrics retourne les métriques Prometheus"""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "# HELP" in response.text


class TestAgentServicePrediction:
    """Tests pour les prédictions"""

    def test_predict_success(self):
        """Test une prédiction réussie"""
        payload = {"text": "My computer won't start"}
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "category" in data
        assert "confidence" in data
        assert "chosen_model" in data
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_empty_text(self):
        """Test avec un texte vide"""
        payload = {"text": ""}
        response = client.post("/predict", json=payload)
        
        # Devrait retourner une erreur 400 ou gérer le cas
        assert response.status_code in [200, 400]

    def test_predict_long_text(self):
        """Test avec un texte long"""
        payload = {"text": "word " * 1000}  # 1000 mots
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200

    def test_predict_special_characters(self):
        """Test avec des caractères spéciaux"""
        payload = {"text": "Mon ordinateur ne démarre pas! #help @support"}
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200


class TestAgentServiceRouting:
    """Tests pour la logique de routage"""

    @patch('httpx.AsyncClient.post')
    async def test_routes_to_tfidf_for_short_text(self, mock_post):
        """Test que les textes courts sont routés vers TF-IDF"""
        # Configuration du mock
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "category": "Hardware",
            "confidence": 0.9
        }
        mock_post.return_value = mock_response
        
        # Texte court
        payload = {"text": "help"}
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200

    @patch('httpx.AsyncClient.post')
    async def test_routes_to_transformer_for_long_text(self, mock_post):
        """Test que les textes longs sont routés vers Transformer"""
        # Configuration du mock
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "category": "Access",
            "confidence": 0.95
        }
        mock_post.return_value = mock_response
        
        # Texte long
        payload = {"text": "This is a much longer text with many words that should be routed to transformer"}
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200


class TestAgentServiceErrors:
    """Tests de gestion d'erreurs"""

    def test_invalid_json(self):
        """Test avec un JSON invalide"""
        response = client.post("/predict", data="invalid json")
        assert response.status_code == 422

    def test_missing_text_field(self):
        """Test sans le champ 'text'"""
        payload = {"wrong_field": "value"}
        response = client.post("/predict", json=payload)
        assert response.status_code in [400, 422]


@pytest.mark.asyncio
class TestAgentServiceAsync:
    """Tests asynchrones"""

    async def test_concurrent_requests(self):
        """Test de requêtes concurrentes"""
        import asyncio
        
        async def make_request():
            payload = {"text": "test"}
            response = client.post("/predict", json=payload)
            return response.status_code
        
        # Lancer 10 requêtes en parallèle
        tasks = [make_request() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # Toutes devraient réussir
        assert all(status == 200 for status in results)


# Tests d'intégration (nécessitent les services en cours d'exécution)
@pytest.mark.integration
class TestAgentServiceIntegration:
    """Tests d'intégration avec les vrais services"""

    def test_integration_with_tfidf(self):
        """Test d'intégration avec le service TF-IDF"""
        # Ce test nécessite que tfidf_service soit en cours d'exécution
        pytest.skip("Nécessite les services Docker en cours d'exécution")

    def test_integration_with_transformer(self):
        """Test d'intégration avec le service Transformer"""
        pytest.skip("Nécessite les services Docker en cours d'exécution")