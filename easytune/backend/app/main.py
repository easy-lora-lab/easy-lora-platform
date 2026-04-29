from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db
from app.routers import datasets, health, model_versions, training_jobs, validation


app = FastAPI(title="EasyTune API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
