from pathlib import Path

from sqlalchemy.orm import Session

from app.models import ModelVersion, TrainingJob


def create_model_version_for_job(db: Session, job: TrainingJob) -> ModelVersion:
    existing = (
        db.query(ModelVersion)
        .filter(ModelVersion.training_job_id == job.id)
        .order_by(ModelVersion.id.desc())
        .first()
    )
    if existing:
        return existing

    export_path = str(Path(job.output_dir or "").resolve() / "export") if job.output_dir else None
    model_version = ModelVersion(
        training_job_id=job.id,
        name=f"{job.name} v1",
        base_model=job.base_model,
        adapter_path=job.output_dir or "",
        export_path=export_path,
        status="ready",
    )
    db.add(model_version)
    db.commit()
    db.refresh(model_version)
    return model_version
