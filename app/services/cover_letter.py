import logging
from app.config import get_settings
from app.models.schemas import CoverLetterRequest, CoverLetterResponse
from app.services.llm import chat_completion

logger = logging.getLogger(__name__)
settings = get_settings()

TONES = {
    "professional": "Polished and formal but warm.",
    "confident": "Bold and direct. No hedging language.",
    "concise": "Maximum 200 words. Every sentence counts.",
}

async def generate_cover_letter(req: CoverLetterRequest) -> CoverLetterResponse:
    tone_instruction = TONES.get(req.tone, TONES["professional"])
    system = f"""You are an expert career coach writing cover letters for senior engineers.
CANDIDATE: {settings.candidate_name} | {settings.candidate_title} | {settings.candidate_years_exp}+ years
SKILLS: {settings.candidate_skills}
TONE: {tone_instruction}
RULES: Never use "I am writing to express my interest". Start with a strong hook.
Max 3 paragraphs. End with a confident closing. Use the actual candidate name. Write in FIRST PERSON. The candidate is writing this letter themselves. Use 'I', not 'Amit' or 'he'."""

    user = f"""Write a cover letter for:
Job: {req.job_title} at {req.company}
Description: {req.job_description}
Write the full letter now. No preamble."""

    logger.info(f"Generating cover letter: {req.job_title} at {req.company}")
    result = await chat_completion(
        messages=[{"role": "user", "content": user}],
        system_prompt=system, temperature=0.7, max_tokens=1000,
    )
    return CoverLetterResponse(
        cover_letter=result.strip(),
        word_count=len(result.split()),
        tone_used=req.tone,
    )
