# api/endpoints/lead_gen.py

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from services.model_service import ModelService

router = APIRouter()


class LeadGenRequest(BaseModel):
    prompt: str


class LeadGenResponse(BaseModel):
    lead: str
    confidence: float  # Added confidence field


def get_model_service():
    return ModelService()


@router.post("/", response_model=LeadGenResponse)
def lead_generation(
    request: LeadGenRequest, model_service: ModelService = Depends(get_model_service)
):
    try:
        result = model_service.generate_lead(request.prompt)
        return LeadGenResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
