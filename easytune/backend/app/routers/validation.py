from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import ValidationRecord
from app.schemas import (
    ValidationGenerateRequest,
    ValidationGenerateResponse,
    ValidationRecordCreate,
    ValidationRecordRead,
)
from app.services.inference_service import generate_validation_answer
from app.services.validation_service import create_validation_record

router = APIRouter(prefix="/api/validation-records", tags=["validation-records"])


@router.post("", response_model=ValidationRecordRead)
def create_record(payload: ValidationRecordCreate, db: Session = Depends(get_db)) -> ValidationRecord:
    return create_validation_record(db, payload)


@router.post("/generate", response_model=ValidationGenerateResponse)
def generate_answer(payload: ValidationGenerateRequest, db: Session = Depends(get_db)) -> ValidationGenerateResponse:
    return generate_validation_answer(db, payload)


@router.get("", response_model=list[ValidationRecordRead])
def list_records(db: Session = Depends(get_db)) -> list[ValidationRecord]:
    return db.query(ValidationRecord).order_by(ValidationRecord.id.desc()).all()
