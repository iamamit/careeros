from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import JobScoreRequest, JobScoreResponse
from app.services.scorer import score_job, get_stream_prompt
from app.services.llm import chat_completion_stream

router = APIRouter(prefix="/score", tags=["Scoring"])

@router.post("", response_model=JobScoreResponse)
async def score(req: JobScoreRequest):
    try:
        return await score_job(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stream")
async def score_stream(req: JobScoreRequest):
    system_prompt, user_prompt = get_stream_prompt(req)
    async def generate():
        async for token in chat_completion_stream(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt,
        ):
            yield token
    return StreamingResponse(generate(), media_type="text/plain")
