# tests/test_unit/test_user_service.py
from unittest.mock import patch
from app.services.user_service import get_user_from_db
from app.models.user import User


@patch("app.services.user_service.get_user_from_db")
def test_get_user_from_db(mock_get_user):
    mock_user = User(username="mockuser", email="mock@example.com")
    mock_get_user.return_value = mock_user

    user = get_user_from_db(1)
    mock_get_user.assert_called_once_with(1)
    assert user.username == "mockuser"
    assert user.email == "mock@example.com"
