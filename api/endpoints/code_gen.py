from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from services.model_service import ModelService

router = APIRouter()


class CodeGenRequest(BaseModel):
    prompt: str


class CodeGenResponse(BaseModel):
    code: str


def get_model_service():
    return ModelService()


@router.post("/", response_model=CodeGenResponse)
def code_generation(
    request: CodeGenRequest, model_service: ModelService = Depends(get_model_service)
):
    try:
        result = model_service.generate_code(request.prompt)
        return CodeGenResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
