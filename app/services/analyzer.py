import logging
from app.config import get_settings
from app.models.schemas import FunctionCallRequest, ExtractedRequirements
from app.services.llm import chat_completion_json

logger = logging.getLogger(__name__)
settings = get_settings()

SYSTEM = """Extract structured requirements from a job description.
Return JSON:
{
  "required_skills": ["skill1"],
  "preferred_skills": ["skill1"],
  "years_experience": <int or null>,
  "work_type": "<remote|hybrid|onsite>",
  "seniority_level": "<junior|mid|senior|lead>",
  "key_responsibilities": ["resp1", "resp2", "resp3"]
}"""

async def extract_job_requirements(req: FunctionCallRequest) -> ExtractedRequirements:
    logger.info("Extracting job requirements")
    raw = await chat_completion_json(
        messages=[{"role": "user", "content": f"Extract from this JD:\n\n{req.job_description}"}],
        system_prompt=SYSTEM, temperature=0.0,
    )
    return ExtractedRequirements(
        required_skills=raw.get("required_skills", []),
        preferred_skills=raw.get("preferred_skills", []),
        years_experience=raw.get("years_experience"),
        work_type=raw.get("work_type", "unknown"),
        seniority_level=raw.get("seniority_level", "unknown"),
        key_responsibilities=raw.get("key_responsibilities", []),
    )
