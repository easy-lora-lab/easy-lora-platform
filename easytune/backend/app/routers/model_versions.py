from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import ModelVersion
from app.schemas import ModelVersionRead

router = APIRouter(prefix="/api/model-versions", tags=["model-versions"])


@router.get("", response_model=list[ModelVersionRead])
def list_model_versions(db: Session = Depends(get_db)) -> list[ModelVersion]:
    return db.query(ModelVersion).order_by(ModelVersion.id.desc()).all()


@router.get("/{model_version_id}", response_model=ModelVersionRead)
def get_model_version(model_version_id: int, db: Session = Depends(get_db)) -> ModelVersion:
    model_version = db.get(ModelVersion, model_version_id)
    if not model_version:
        raise HTTPException(status_code=404, detail="Model version not found.")
    return model_version
