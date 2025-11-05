from fastapi.testclient import TestClient
from streamlit import json
from app import app, models

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    json = response.json()
    assert json["message"] == "IRIS Classification API"

def health_check():
    response = client.get("/health")
    assert response.status_code == 200
    json = response.json()
    assert json == {"status": "healthy"}

def test_models_loaded():
    # Confirm at least one model loaded
    assert len(models) > 0

def test_single_prediction():
    payload = {
        "features": {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        },
        "model_name": "random_forest"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "species" in data
    assert "model_used" in data


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    
    # Response will be in plain text format
    assert isinstance(response.text, str)
    
    # We check that at least one expected metric label is present
    assert "iris_predictions_total" in response.text
    assert "iris_models_loaded" in response.text
