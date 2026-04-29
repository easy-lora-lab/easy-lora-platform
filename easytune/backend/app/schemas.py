from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class DatasetRead(ORMModel):
    id: int
    name: str
    original_file_path: str
    converted_file_path: str | None
    file_type: str
    file_size: int
    sample_count: int
    formatting: str
    quality_score: float
    report_json: dict[str, Any] | None
    dataset_info_json: dict[str, Any] | None
    status: str
    created_at: datetime


class DatasetInfoResponse(BaseModel):
    dataset_id: int
    dataset_name: str
    dataset_info_json: dict[str, Any] | None


class DatasetPreviewResponse(BaseModel):
    dataset_id: int
    records: list[dict[str, Any]]
    total_records: int


class DatasetSplitRequest(BaseModel):
    valid_ratio: float = Field(default=0.1, ge=0.01, le=0.5)
    seed: int = 42


class DatasetSplitResponse(BaseModel):
    dataset_id: int
    train_path: str
    valid_path: str
    train_count: int
    valid_count: int


class TrainingJobCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    dataset_id: int
    model_family: str = "qwen"
    base_model: str = Field(min_length=1)
    template: str = "default"
    stage: str = "sft"
    finetuning_type: str = "lora"
    budget_level: str = "balanced"


class TrainingJobRead(ORMModel):
    id: int
    name: str
    dataset_id: int
    model_family: str
    base_model: str
    template: str
    stage: str
    finetuning_type: str
    budget_level: str
    config_path: str | None
    output_dir: str | None
    status: str
    progress: int
    command: str | None
    log_path: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class TrainingLogsResponse(BaseModel):
    job_id: int
    log_path: str | None
    content: str


class ModelVersionRead(ORMModel):
    id: int
    training_job_id: int
    name: str
    base_model: str
    adapter_path: str
    export_path: str | None
    status: str
    created_at: datetime


class ValidationRecordCreate(BaseModel):
    model_version_id: int
    prompt: str = Field(min_length=1)
    expected_answer: str | None = None
    actual_answer: str = Field(min_length=1)
    human_score: int = Field(ge=1, le=5)
    human_note: str | None = None


class ValidationGenerateRequest(BaseModel):
    model_version_id: int
    prompt: str = Field(min_length=1)
    system_prompt: str | None = None
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=512, ge=1, le=8192)


class ValidationGenerateResponse(BaseModel):
    model_version_id: int
    provider: str
    model: str
    actual_answer: str
    raw_response: dict[str, Any] | None = None


class ValidationRecordRead(ORMModel):
    id: int
    model_version_id: int
    prompt: str
    expected_answer: str | None
    actual_answer: str
    human_score: int
    human_note: str | None
    created_at: datetime
