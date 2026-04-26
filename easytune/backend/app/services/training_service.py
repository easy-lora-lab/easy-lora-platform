import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import Dataset, TrainingJob
from app.schemas import TrainingJobCreate
from app.services.llamafactory_config_service import generate_training_yaml
from app.services.log_service import append_log
from app.services.model_service import create_model_version_for_job


VALID_MODEL_FAMILIES = {"qwen", "rwkv"}
VALID_TEMPLATES = {"qwen", "rwkv", "default"}
VALID_FINETUNING_TYPES = {"lora", "freeze", "full"}
VALID_BUDGETS = {"low", "balanced", "high"}


def create_training_job(db: Session, payload: TrainingJobCreate) -> TrainingJob:
    dataset = db.get(Dataset, payload.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    if dataset.status != "converted":
        raise HTTPException(status_code=400, detail="Please convert the dataset before creating a training job.")
    if payload.model_family not in VALID_MODEL_FAMILIES:
        raise HTTPException(status_code=400, detail="model_family must be qwen or rwkv.")
    if payload.template not in VALID_TEMPLATES:
        raise HTTPException(status_code=400, detail="template must be qwen, rwkv or default.")
    if payload.stage != "sft":
        raise HTTPException(status_code=400, detail="Only sft stage is supported in v1.")
    if payload.finetuning_type not in VALID_FINETUNING_TYPES:
        raise HTTPException(status_code=400, detail="finetuning_type must be lora, freeze or full.")
    if payload.budget_level not in VALID_BUDGETS:
        raise HTTPException(status_code=400, detail="budget_level must be low, balanced or high.")

    job = TrainingJob(**payload.model_dump(), status="pending", progress=0)
    db.add(job)
    db.flush()
    paths = generate_training_yaml(job, dataset)
    job.config_path = paths["config_path"]
    job.output_dir = paths["output_dir"]
    job.log_path = paths["log_path"]
    job.command = paths["command"]
    db.commit()
    db.refresh(job)
    return job


def mark_job_running(db: Session, job_id: int) -> TrainingJob:
    job = db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found.")
    if job.status == "running":
        raise HTTPException(status_code=400, detail="Training job is already running.")
    if job.status == "completed":
        raise HTTPException(status_code=400, detail="Training job is already completed.")
    job.status = "running"
    job.progress = 0
    job.error_message = None
    job.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(job)
    append_log(job.log_path, f"[{datetime.utcnow().isoformat()}] Job queued.")
    return job


def run_training_job(job_id: int) -> None:
    db = SessionLocal()
    try:
        job = db.get(TrainingJob, job_id)
        if not job:
            return
        cli_path = shutil.which("llamafactory-cli")
        if job.model_family == "qwen" and cli_path:
            _run_real_training(db, job, cli_path)
        else:
            _run_mock_training(db, job)
    except Exception as exc:  # noqa: BLE001
        job = db.get(TrainingJob, job_id)
        if job:
            job.status = "failed"
            job.error_message = str(exc)
            job.updated_at = datetime.utcnow()
            db.commit()
            append_log(job.log_path, f"[ERROR] {exc}")
    finally:
        db.close()


def _run_real_training(db: Session, job: TrainingJob, cli_path: str) -> None:
    command = [cli_path, "train", job.config_path or ""]
    job.command = " ".join(command)
    job.status = "running"
    job.progress = 1
    job.updated_at = datetime.utcnow()
    db.commit()
    append_log(job.log_path, f"[{datetime.utcnow().isoformat()}] Starting real training: {job.command}")

    with Path(job.log_path or "").open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
        while process.poll() is None:
            job.progress = min(95, job.progress + 1)
            job.updated_at = datetime.utcnow()
            db.commit()
            time.sleep(5)
        return_code = process.returncode

    if return_code == 0:
        _complete_job(db, job)
    else:
        job.status = "failed"
        job.error_message = f"llamafactory-cli exited with code {return_code}"
        job.updated_at = datetime.utcnow()
        db.commit()
        append_log(job.log_path, f"[ERROR] {job.error_message}")


def _run_mock_training(db: Session, job: TrainingJob) -> None:
    if job.model_family == "rwkv":
        append_log(job.log_path, "[mock] RWKV finetuning runner placeholder. Falling back to mock train.")
    else:
        append_log(job.log_path, "[mock] llamafactory-cli not found. Falling back to mock train.")
    steps = [
        (10, "loading dataset"),
        (25, "building trainer"),
        (45, "training epoch 1"),
        (65, "saving adapter"),
        (85, "writing metadata"),
        (100, "completed"),
    ]
    for progress, message in steps:
        append_log(job.log_path, f"[mock] {message}")
        job.status = "running"
        job.progress = progress
        job.updated_at = datetime.utcnow()
        db.commit()
        time.sleep(0.6)

    output_dir = Path(job.output_dir or "")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "mock": True,
                "model_family": job.model_family,
                "base_model": job.base_model,
                "finetuning_type": job.finetuning_type,
                "created_at": datetime.utcnow().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "README.txt").write_text(
        "Mock EasyTune training output. Replace with real LLaMA-Factory adapter files in production.\n",
        encoding="utf-8",
    )
    _complete_job(db, job)


def _complete_job(db: Session, job: TrainingJob) -> None:
    job.status = "completed"
    job.progress = 100
    job.updated_at = datetime.utcnow()
    db.commit()
    append_log(job.log_path, f"[{datetime.utcnow().isoformat()}] Training completed.")
    create_model_version_for_job(db, job)
