import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.models.schemas import HealthResponse
from app.routers import score, generate

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.groq_api_key:
        logging.warning("GROQ_API_KEY not set — get free key at console.groq.com")
    yield

app = FastAPI(title="CareerOS — Week 1", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(score.router)
app.include_router(generate.router)

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="ok" if settings.groq_api_key else "degraded — add GROQ_API_KEY to .env",
        model=settings.groq_model, env=settings.app_env,
    )

@app.get("/", tags=["System"])
async def root():
    return {"service": "CareerOS Week 1", "docs": "/docs", "health": "/health"}
