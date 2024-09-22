# tests/test_integration/test_endpoints.py
import pytest
from fastapi.testclient import TestClient
from app.main import app  # Ensure your FastAPI app is instantiated in app/main.py
from app.database import Base, engine, SessionLocal
from app.models.user import User

client = TestClient(app)


# Create a new database for testing
@pytest.fixture(scope="module")
def test_db():
    # Create the tables
    Base.metadata.create_all(bind=engine)
    yield
    # Drop the tables after tests
    Base.metadata.drop_all(bind=engine)


# Dependency override for testing
@pytest.fixture(scope="module")
def client(test_db):
    with TestClient(app) as c:
        yield c


def test_create_user(client):
    response = client.post(
        "/users/", json={"username": "testuser", "email": "test@example.com"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "testuser"
    assert data["email"] == "test@example.com"
    assert "id" in data


def test_create_user_duplicate_email(client):
    # First creation
    client.post(
        "/users/", json={"username": "testuser1", "email": "duplicate@example.com"}
    )
    # Attempt to create another user with the same email
    response = client.post(
        "/users/", json={"username": "testuser2", "email": "duplicate@example.com"}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Email already registered"

    # tests/test_integration/test_endpoints.py


def test_get_user(client):
    # First, create a user
    response = client.post(
        "/users/", json={"username": "getuser", "email": "getuser@example.com"}
    )
    assert response.status_code == 200
    user_data = response.json()
    user_id = user_data["id"]

    # Now, retrieve the user by ID
    response = client.get(f"/users/{user_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "getuser"
    assert data["email"] == "getuser@example.com"
    assert data["id"] == user_id


def test_get_user_not_found(client):
    response = client.get("/users/99999")  # Assuming this ID does not exist
    assert response.status_code == 404
    data = response.json()
    assert data["detail"] == "User not found"
