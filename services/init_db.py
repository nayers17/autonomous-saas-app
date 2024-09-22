# services/init_db.py

from services.database import engine, Base
from models.feedback_model import Feedback


def create_tables():
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")


if __name__ == "__main__":
    create_tables()
