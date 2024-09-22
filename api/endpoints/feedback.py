# api/endpoints/feedback.py

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from models.feedback_model import Feedback
from services.database import SessionLocal
from utils.logger import get_logger

router = APIRouter()
logger = get_logger("feedback")


class FeedbackRequest(BaseModel):
    service: str
    input: str
    output: str  # Treat output as a string for consistency
    feedback: str


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/feedback/")
def submit_feedback(request: FeedbackRequest, db: Session = Depends(get_db)):
    try:
        feedback_entry = Feedback(
            service=request.service,
            input=request.input,
            output=request.output,  # Already a string, no need to convert
            feedback=request.feedback,
        )
        db.add(feedback_entry)
        db.commit()
        db.refresh(feedback_entry)
        logger.info(f"Feedback Stored: {feedback_entry}")
        return {"message": "Feedback submitted successfully."}
    except Exception as e:
        logger.error(f"Feedback Submission Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
