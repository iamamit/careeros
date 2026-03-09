from fastapi import APIRouter, HTTPException
from app.models.schemas import CoverLetterRequest, CoverLetterResponse, FunctionCallRequest, ExtractedRequirements
from app.services.cover_letter import generate_cover_letter
from app.services.analyzer import extract_job_requirements

router = APIRouter(prefix="/generate", tags=["Generation"])

@router.post("/cover-letter", response_model=CoverLetterResponse)
async def cover_letter(req: CoverLetterRequest):
    try:
        return await generate_cover_letter(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/extract", response_model=ExtractedRequirements)
async def extract(req: FunctionCallRequest):
    try:
        return await extract_job_requirements(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
