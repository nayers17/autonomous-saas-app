import sys
import os
from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base, SessionLocal

# Add the project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session", autouse=True)
def set_pythonpath():
    os.environ["PYTHONPATH"] = str(PROJECT_ROOT)


# Set up the test database
@pytest.fixture(scope="session")
def engine():
    test_engine = create_engine(
        "postgresql://superchillpogger:Asdfqwer1234!@localhost/autonomous_saas_test"
    )
    Base.metadata.create_all(bind=test_engine)
    return test_engine


# Set up a session for tests
@pytest.fixture(scope="function")
def session(engine):
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = Session()
    yield session
    session.close()


# Override the SessionLocal for tests
@pytest.fixture(scope="function")
def override_session(session):
    def _override_session():
        return session

    return _override_session
