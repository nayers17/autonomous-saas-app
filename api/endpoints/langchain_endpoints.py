# api/endpoints/langchain_endpoints.py

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from services.langchain_service import LangChainService
from utils.logger import get_logger

router = APIRouter()
logger = get_logger("langchain")


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float


class CodeRequest(BaseModel):
    prompt: str


class CodeResponse(BaseModel):
    code: str


class LeadRequest(BaseModel):
    prompt: str


class LeadResponse(BaseModel):
    lead: str
    confidence: float


def get_langchain_service():
    return LangChainService()


@router.post("/sentiment-analysis/", response_model=SentimentResponse)
def sentiment_analysis(
    request: SentimentRequest,
    service: LangChainService = Depends(get_langchain_service),
):
    try:
        logger.info(f"Sentiment Analysis Request: {request.text}")
        result = service.analyze_sentiment(request.text)
        # Assuming the pipeline returns a dictionary with 'label' and 'score'
        sentiment = result.get("label", "UNKNOWN")
        confidence = float(result.get("score", 0.0))
        logger.info(f"Sentiment Analysis Result: {sentiment}, Confidence: {confidence}")
        return SentimentResponse(sentiment=sentiment, confidence=confidence)
    except Exception as e:
        logger.error(f"Sentiment Analysis Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post("/code-generation/", response_model=CodeResponse)
def code_generation(
    request: CodeRequest, service: LangChainService = Depends(get_langchain_service)
):
    try:
        logger.info(f"Code Generation Request: {request.prompt}")
        result = service.generate_code(request.prompt)
        code = (
            result[0]["generated_text"].strip()
            if isinstance(result, list)
            else result.strip()
        )
        logger.info(f"Code Generation Result: {code}")
        return CodeResponse(code=code)
    except Exception as e:
        logger.error(f"Code Generation Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post("/lead-generation/", response_model=LeadResponse)
def lead_generation(
    request: LeadRequest, service: LangChainService = Depends(get_langchain_service)
):
    try:
        logger.info(f"Lead Generation Request: {request.prompt}")
        result = service.generate_lead(request.prompt)
        lead = (
            result[0]["generated_text"].strip()
            if isinstance(result, list)
            else result.strip()
        )
        confidence = 0.90  # As per your dummy confidence score
        logger.info(f"Lead Generation Result: {lead}, Confidence: {confidence}")
        return LeadResponse(lead=lead, confidence=confidence)
    except Exception as e:
        logger.error(f"Lead Generation Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
