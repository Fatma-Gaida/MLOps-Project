import requests
import json

# Configuration
BASE_URL = "http://localhost:8001"

def test_health():
    """Test du endpoint /health"""
    print("\n=== Testing /health ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_predict():
    """Test du endpoint /predict"""
    print("\n=== Testing /predict ===")
    
    test_cases = [
        "My laptop screen is broken and needs repair",
        "I cannot access my email account",
        "Need to order new keyboards for the office",
        "Server is down and not responding",
        "Request for additional storage space"
    ]
    
    for text in test_cases:
        print(f"\n Text: {text}")
        
        payload = {"text": text}
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f" Predicted: {result['predicted_label']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Processing Time: {result['processing_time']:.3f}s")
        else:
            print(f" Error: {response.status_code}")
            print(f"   {response.text}")

def test_metrics():
    """Test du endpoint /metrics"""
    print("\n=== Testing /metrics ===")
    response = requests.get(f"{BASE_URL}/metrics")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        # Afficher les premières lignes des métriques
        lines = response.text.split('\n')[:20]
        print('\n'.join(lines))
        print("...")
    
    return response.status_code == 200

def test_root():
    """Test du endpoint root"""
    print("\n=== Testing / (root) ===")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

if __name__ == "__main__":
    print(" Starting API Tests...")
    print(f" Base URL: {BASE_URL}")
    
    try:
        # Test tous les endpoints
        health_ok = test_health()
        root_ok = test_root()
        
        if health_ok:
            test_predict()
            test_metrics()
            print("\n All tests completed!")
        else:
            print("\n Health check failed. Service may not be ready.")
            
    except requests.exceptions.ConnectionError:
        print(f"\n Cannot connect to {BASE_URL}")
        print("Make sure the service is running:")
        print("  python main.py")
        print("  or")
        print("  uvicorn main:app --reload --port 8001")
    except Exception as e:
        print(f"\n Error: {e}")