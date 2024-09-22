# tests/test_unit/test_user.py
from app.models.user import User


def test_get_domain():
    user = User(username="john_doe", email="john@example.com")
    assert user.get_domain() == "example.com"

    user.email = "jane@anotherdomain.org"
    assert user.get_domain() == "anotherdomain.org"
