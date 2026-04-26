from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    original_file_path: Mapped[str] = mapped_column(Text, nullable=False)
    converted_file_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_type: Mapped[str] = mapped_column(String(32), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    formatting: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    quality_score: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    report_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    dataset_info_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="uploaded")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    training_jobs: Mapped[list["TrainingJob"]] = relationship(back_populates="dataset")


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    model_family: Mapped[str] = mapped_column(String(32), nullable=False, default="qwen")
    base_model: Mapped[str] = mapped_column(Text, nullable=False)
    template: Mapped[str] = mapped_column(String(64), nullable=False, default="default")
    stage: Mapped[str] = mapped_column(String(64), nullable=False, default="sft")
    finetuning_type: Mapped[str] = mapped_column(String(64), nullable=False, default="lora")
    budget_level: Mapped[str] = mapped_column(String(64), nullable=False, default="balanced")
    config_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_dir: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    command: Mapped[str | None] = mapped_column(Text, nullable=True)
    log_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    dataset: Mapped[Dataset] = relationship(back_populates="training_jobs")
    model_versions: Mapped[list["ModelVersion"]] = relationship(back_populates="training_job")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    training_job_id: Mapped[int] = mapped_column(ForeignKey("training_jobs.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    base_model: Mapped[str] = mapped_column(Text, nullable=False)
    adapter_path: Mapped[str] = mapped_column(Text, nullable=False)
    export_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ready")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    training_job: Mapped[TrainingJob] = relationship(back_populates="model_versions")
    validation_records: Mapped[list["ValidationRecord"]] = relationship(back_populates="model_version")


class ValidationRecord(Base):
    __tablename__ = "validation_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    expected_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    actual_answer: Mapped[str] = mapped_column(Text, nullable=False)
    human_score: Mapped[int] = mapped_column(Integer, nullable=False)
    human_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    model_version: Mapped[ModelVersion] = relationship(back_populates="validation_records")
