import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse

from app.database import init_db
from app.routers import datasets, health, model_versions, training_jobs, validation


app = FastAPI(title="EasyTune API", version="0.1.0")


def _cors_origins() -> list[str]:
    raw_value = os.getenv("EASYTUNE_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    return [origin.strip() for origin in raw_value.split(",") if origin.strip()]


cors_origins = _cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def require_api_token(request: Request, call_next):
    api_token = os.getenv("EASYTUNE_API_TOKEN")
    if api_token and request.url.path.startswith("/api") and request.method not in {"GET", "HEAD", "OPTIONS"}:
        expected_header = f"Bearer {api_token}"
        if request.headers.get("authorization") != expected_header:
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API token."})
    return await call_next(request)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


app.include_router(health.router)
app.include_router(datasets.router)
app.include_router(training_jobs.router)
app.include_router(model_versions.router)
app.include_router(validation.router)


@app.get("/")
def root() -> dict[str, object]:
    return {
        "name": "EasyTune API",
        "version": app.version,
        "docs": "/docs",
        "health": "/api/health",
        "routes": {
            "datasets": "/api/datasets",
            "training_jobs": "/api/training-jobs",
            "model_versions": "/api/model-versions",
            "validation_records": "/api/validation-records",
            "validation_generate": "/api/validation-records/generate",
        },
    }


@app.get("/api")
def api_index() -> dict[str, object]:
    return root()
