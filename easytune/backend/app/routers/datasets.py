from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Dataset
from app.schemas import (
    DatasetInfoResponse,
    DatasetPreviewResponse,
    DatasetRead,
    DatasetSplitRequest,
    DatasetSplitResponse,
)
from app.services.dataset_service import (
    convert_dataset,
    create_dataset_from_upload,
    dataset_name,
    preview_dataset,
    split_dataset,
)

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.post("/upload", response_model=DatasetRead)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str | None = Form(default=None),
    db: Session = Depends(get_db),
) -> Dataset:
    return await create_dataset_from_upload(db, file, name)


@router.get("", response_model=list[DatasetRead])
def list_datasets(db: Session = Depends(get_db)) -> list[Dataset]:
    return db.query(Dataset).order_by(Dataset.id.desc()).all()


@router.get("/{dataset_id}", response_model=DatasetRead)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)) -> Dataset:
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Dataset not found.")
    return dataset


@router.post("/{dataset_id}/convert", response_model=DatasetRead)
def convert(dataset_id: int, db: Session = Depends(get_db)) -> Dataset:
    return convert_dataset(db, dataset_id)


@router.get("/{dataset_id}/preview", response_model=DatasetPreviewResponse)
def preview(dataset_id: int, limit: int = 8, db: Session = Depends(get_db)) -> DatasetPreviewResponse:
    records, total_records = preview_dataset(db, dataset_id, limit)
    return DatasetPreviewResponse(dataset_id=dataset_id, records=records, total_records=total_records)


@router.post("/{dataset_id}/split", response_model=DatasetSplitResponse)
def split(dataset_id: int, payload: DatasetSplitRequest, db: Session = Depends(get_db)) -> DatasetSplitResponse:
    result = split_dataset(db, dataset_id, payload.valid_ratio, payload.seed)
    return DatasetSplitResponse(**result)


@router.get("/{dataset_id}/dataset-info", response_model=DatasetInfoResponse)
def get_dataset_info(dataset_id: int, db: Session = Depends(get_db)) -> DatasetInfoResponse:
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Dataset not found.")
    return DatasetInfoResponse(
        dataset_id=dataset.id,
        dataset_name=dataset_name(dataset.id),
        dataset_info_json=dataset.dataset_info_json,
    )
