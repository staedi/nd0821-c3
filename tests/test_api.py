from fastapi.testclient import TestClient
from main import app
import pytest

@pytest.fixture
def client():
    with TestClient(app) as cli:
        yield cli
#     client = TestClient(app)
#     return client

# GET: Welcome message
def test_welcome(client):
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {'msg':'Welcome!'}

# POST: payload with over 50K input
def test_predict_over(client, payload_over_sample):
    response = client.post('/predict', json=payload_over_sample)
    assert response.status_code == 200
    assert response.json()['result'] == 1

# POST: payload with under 50K input
def test_predict_under(client, payload_under_sample):
    response = client.post('/predict', json=payload_under_sample)
    assert response.status_code == 200
    assert response.json()['result'] == 0

# POST: Errorneous input
def test_predict_error(client, payload_error_sample):
    response = client.post('/predict', json=payload_error_sample)
    assert response.status_code != 200
