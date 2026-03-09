import logging
from app.config import get_settings
from app.models.schemas import JobScoreRequest, JobScoreResponse, SkillMatch
from app.services.llm import chat_completion_json

logger = logging.getLogger(__name__)
settings = get_settings()

def _system_prompt():
    return f"""You are a senior technical recruiter evaluating job fit.
CANDIDATE: {settings.candidate_name} | {settings.candidate_title} | {settings.candidate_years_exp}+ years
SKILLS: {settings.candidate_skills}
Be honest and precise. Do not inflate scores."""

def _score_prompt(req: JobScoreRequest):
    return f"""Evaluate this job for the candidate.
JOB: {req.job_title} at {req.company} ({req.location})
{f'Salary: {req.salary}' if req.salary else ''}
DESCRIPTION: {req.job_description}

Return JSON:
{{
  "match_score": <0-100>,
  "verdict": "<one sentence>",
  "matched_skills": [{{"skill": "", "present": true, "note": ""}}],
  "missing_skills": [{{"skill": "", "present": false, "note": ""}}],
  "strengths": ["", "", ""],
  "gaps": ["", "", ""],
  "interview_prep": ["", "", ""],
  "apply_recommendation": "<STRONG_YES|YES|MAYBE|NO>",
  "salary_comment": "<comment or null>"
}}"""

async def score_job(req: JobScoreRequest) -> JobScoreResponse:
    logger.info(f"Scoring: {req.job_title} at {req.company}")
    raw = await chat_completion_json(
        messages=[{"role": "user", "content": _score_prompt(req)}],
        system_prompt=_system_prompt(), temperature=0.1,
    )
    return JobScoreResponse(
        job_title=req.job_title, company=req.company,
        match_score=raw["match_score"], verdict=raw["verdict"],
        matched_skills=[SkillMatch(**s) for s in raw.get("matched_skills", [])],
        missing_skills=[SkillMatch(**s) for s in raw.get("missing_skills", [])],
        strengths=raw.get("strengths", []), gaps=raw.get("gaps", []),
        interview_prep=raw.get("interview_prep", []),
        apply_recommendation=raw["apply_recommendation"],
        salary_comment=raw.get("salary_comment"),
    )

def get_stream_prompt(req: JobScoreRequest):
    user = f"""Evaluate this job:
JOB: {req.job_title} at {req.company} ({req.location})
{req.job_description}

Write evaluation with sections:
1. Match Score: X/100 — verdict
2. Why You're a Strong Fit: 3 bullets
3. Gaps to Address: up to 3 bullets
4. Interview Prep: 3 topics
5. Recommendation: STRONG_YES/YES/MAYBE/NO — why"""
    return _system_prompt(), user
