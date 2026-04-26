from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import TrainingJob
from app.schemas import TrainingJobCreate, TrainingJobRead, TrainingLogsResponse
from app.services.log_service import read_log
from app.services.training_service import create_training_job, mark_job_running, run_training_job

router = APIRouter(prefix="/api/training-jobs", tags=["training-jobs"])


@router.post("", response_model=TrainingJobRead)
def create_job(payload: TrainingJobCreate, db: Session = Depends(get_db)) -> TrainingJob:
    return create_training_job(db, payload)


@router.get("", response_model=list[TrainingJobRead])
def list_jobs(db: Session = Depends(get_db)) -> list[TrainingJob]:
    return db.query(TrainingJob).order_by(TrainingJob.id.desc()).all()


@router.get("/{job_id}", response_model=TrainingJobRead)
def get_job(job_id: int, db: Session = Depends(get_db)) -> TrainingJob:
    job = db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found.")
    return job


@router.post("/{job_id}/start", response_model=TrainingJobRead)
def start_job(job_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)) -> TrainingJob:
    job = mark_job_running(db, job_id)
    background_tasks.add_task(run_training_job, job_id)
    return job


@router.get("/{job_id}/logs", response_model=TrainingLogsResponse)
def get_job_logs(job_id: int, db: Session = Depends(get_db)) -> TrainingLogsResponse:
    job = db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found.")
    return TrainingLogsResponse(job_id=job.id, log_path=job.log_path, content=read_log(job.log_path))
