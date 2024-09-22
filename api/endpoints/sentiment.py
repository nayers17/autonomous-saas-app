from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from services.model_service import ModelService

router = APIRouter()


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float


def get_model_service():
    return ModelService()


@router.post("/", response_model=SentimentResponse)
def sentiment_analysis(
    request: SentimentRequest, model_service: ModelService = Depends(get_model_service)
):
    try:
        result = model_service.analyze_sentiment(request.text)
        return SentimentResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
