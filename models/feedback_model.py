# models/feedback_model.py

from sqlalchemy import Column, Integer, String, Text
from services.database import Base


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    service = Column(String, index=True)
    input = Column(Text)
    output = Column(Text)
    feedback = Column(Text)
