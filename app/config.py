from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-70b-versatile"
    app_env: str = "development"
    log_level: str = "INFO"
    candidate_name: str = "Amit Gautam"
    candidate_title: str = "Senior Backend Engineer"
    candidate_skills: str = (
        "Java 17, Spring Boot, Python, FastAPI, Kafka, PostgreSQL, "
        "MongoDB, GCP, Docker, Camunda 7, BPMN 2.0, React.js, "
        "GitHub Actions, GitLab CI"
    )
    candidate_years_exp: int = 7

    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache
def get_settings() -> Settings:
    return Settings()
