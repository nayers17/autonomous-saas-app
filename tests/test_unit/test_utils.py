# tests/test_unit/test_utils.py
from app.utils import add


def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0
