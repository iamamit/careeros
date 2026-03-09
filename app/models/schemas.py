from pydantic import BaseModel, Field
from typing import Optional

class JobScoreRequest(BaseModel):
    job_title: str = Field(..., example="Senior Backend Engineer")
    company: str = Field(..., example="Capco")
    location: str = Field(..., example="Edinburgh, UK")
    job_description: str = Field(..., min_length=10, example="We are looking for a Java engineer...")
    salary: Optional[str] = Field(None, example="80,000 - 100,000")

class CoverLetterRequest(BaseModel):
    job_title: str
    company: str
    job_description: str
    tone: str = Field("professional", example="professional")

class FunctionCallRequest(BaseModel):
    job_description: str

class SkillMatch(BaseModel):
    skill: str
    present: bool
    note: str

class JobScoreResponse(BaseModel):
    job_title: str
    company: str
    match_score: int = Field(..., ge=0, le=100)
    verdict: str
    matched_skills: list[SkillMatch]
    missing_skills: list[SkillMatch]
    strengths: list[str]
    gaps: list[str]
    interview_prep: list[str]
    apply_recommendation: str
    salary_comment: Optional[str] = None

class CoverLetterResponse(BaseModel):
    cover_letter: str
    word_count: int
    tone_used: str

class ExtractedRequirements(BaseModel):
    required_skills: list[str]
    preferred_skills: list[str]
    years_experience: Optional[int]
    work_type: str
    seniority_level: str
    key_responsibilities: list[str]

class HealthResponse(BaseModel):
    status: str
    model: str
    env: str
