from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models import ModelVersion, ValidationRecord
from app.schemas import ValidationRecordCreate


def create_validation_record(db: Session, payload: ValidationRecordCreate) -> ValidationRecord:
    model_version = db.get(ModelVersion, payload.model_version_id)
    if not model_version:
        raise HTTPException(status_code=404, detail="Model version not found.")
    record = ValidationRecord(**payload.model_dump())
    db.add(record)
    db.commit()
    db.refresh(record)
    return record
