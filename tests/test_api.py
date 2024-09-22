from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is running"}


def test_get_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "Everything is running smoothly!"}


def test_sentiment_analysis():
    response = client.post("/sentiment-analysis/", json={"text": "I love this!"})
    assert response.status_code == 200
    assert response.json()["sentiment"] in ["POSITIVE", "NEGATIVE"]
    assert isinstance(response.json()["confidence"], float)


def test_lead_generation():
    response = client.post(
        "/lead-generation/", json={"prompt": "Generate a new marketing lead."}
    )
    assert response.status_code == 200
    assert isinstance(response.json()["lead"], str)
    assert isinstance(response.json()["confidence"], float)


def test_code_generation():
    response = client.post(
        "/code-generation/", json={"prompt": "Write a function that says hello."}
    )
    assert response.status_code == 200
    assert isinstance(response.json()["code"], str)
