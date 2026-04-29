import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from sqlalchemy.orm import Session
import yaml

from app.database import APP_DIR, SessionLocal
from app.models import Dataset, TrainingJob
from app.schemas import TrainingJobCreate
from app.services.llamafactory_config_service import generate_training_yaml
from app.services.log_service import append_log, read_log
from app.services.model_service import create_model_version_for_job


VALID_MODEL_FAMILIES = {"qwen", "rwkv"}
VALID_TEMPLATES = {"qwen", "rwkv", "default"}
VALID_FINETUNING_TYPES = {"lora", "qlora", "freeze", "full"}
VALID_BUDGETS = {"low", "balanced", "high"}
VENDOR_ROOT = APP_DIR / "vendor"
RWKV_PEFT_ROOT = VENDOR_ROOT / "rwkv_peft"
LLAMAFACTORY_ROOT = VENDOR_ROOT / "llamafactory"
FAILURE_HINTS = [
    ("cuda out of memory", "GPU 显存不足。请降低预算等级、cutoff_len 或 batch size。"),
    ("outofmemoryerror", "GPU 显存不足。请降低预算等级、cutoff_len 或 batch size。"),
    ("no module named", "训练环境缺少 Python 依赖。请检查 LLaMA-Factory/RWKV runner 环境。"),
    ("permission denied", "训练进程没有读写某个路径的权限。请检查模型、数据集和输出目录权限。"),
    ("file not found", "训练进程找不到需要的文件。请检查 base_model、数据集转换文件和配置路径。"),
    ("no such file or directory", "训练进程找不到需要的文件。请检查 base_model、数据集转换文件和配置路径。"),
    ("failed to load", "模型或数据集加载失败。请检查 base_model、template 和数据格式。"),
    ("template", "训练模板配置可能不匹配。请检查 template 与模型家族。"),
]


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
        raise HTTPException(status_code=400, detail="finetuning_type must be lora, qlora, freeze or full.")
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
    runner = _find_training_runner(job)
    _run_preflight_checks(db, job, runner)
    job.status = "running"
    job.progress = 0
    job.error_message = None
    job.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(job)
    append_log(job.log_path, f"[{datetime.utcnow().isoformat()}] Job queued.")
    if runner:
        append_log(job.log_path, f"[preflight] Runner found: {runner['label']}")
    else:
        append_log(job.log_path, f"[preflight] Runner not found for {job.model_family}; mock train will be used.")
    return job


def run_training_job(job_id: int) -> None:
    db = SessionLocal()
    try:
        job = db.get(TrainingJob, job_id)
        if not job:
            return
        runner = _find_training_runner(job)
        if runner:
            _run_real_training(db, job, runner)
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


def _run_real_training(db: Session, job: TrainingJob, runner: dict[str, Any]) -> None:
    command = _build_training_command(job, runner)
    job.command = " ".join(command)
    job.status = "running"
    job.progress = 1
    job.updated_at = datetime.utcnow()
    db.commit()
    append_log(job.log_path, f"[{datetime.utcnow().isoformat()}] Starting real training: {job.command}")

    with Path(job.log_path or "").open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=runner.get("cwd"),
            env=runner.get("env"),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
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
        job.error_message = _explain_training_failure(job.log_path, f"{job.model_family} runner exited with code {return_code}")
        job.updated_at = datetime.utcnow()
        db.commit()
        append_log(job.log_path, f"[ERROR] {job.error_message}")


def _run_mock_training(db: Session, job: TrainingJob) -> None:
    if job.model_family == "rwkv":
        append_log(job.log_path, "[mock] rwkv-finetune not found. Falling back to mock train.")
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
    model_version = create_model_version_for_job(db, job)
    append_log(
        job.log_path,
        f"[{datetime.utcnow().isoformat()}] ModelVersion {model_version.id} created with status={model_version.status}.",
    )


def _find_training_runner(job: TrainingJob) -> dict[str, Any] | None:
    if job.model_family == "rwkv":
        cli_path = shutil.which("rwkv-finetune")
        if cli_path:
            return {"label": cli_path, "kind": "rwkv_cli", "cli_path": cli_path}
        if (RWKV_PEFT_ROOT / "train.py").exists():
            return {
                "label": f"vendor RWKV-PEFT ({RWKV_PEFT_ROOT})",
                "kind": "rwkv_peft_vendor",
                "cwd": str(RWKV_PEFT_ROOT),
                "env": _env_with_pythonpath(str(RWKV_PEFT_ROOT)),
            }
        return None

    cli_path = shutil.which("llamafactory-cli")
    if cli_path:
        return {"label": cli_path, "kind": "llamafactory_cli", "cli_path": cli_path}
    src_path = LLAMAFACTORY_ROOT / "src"
    if (src_path / "llamafactory").exists():
        return {
            "label": f"vendor LLaMA-Factory ({LLAMAFACTORY_ROOT})",
            "kind": "llamafactory_vendor",
            "cwd": str(LLAMAFACTORY_ROOT),
            "env": _env_with_pythonpath(str(src_path)),
        }
    return None


def _build_training_command(job: TrainingJob, runner: dict[str, Any]) -> list[str]:
    kind = runner["kind"]
    if kind == "rwkv_cli":
        return [runner["cli_path"], "train", job.config_path or ""]
    if kind == "rwkv_peft_vendor":
        return _build_rwkv_peft_command(job)
    if kind == "llamafactory_vendor":
        return [sys.executable, "-m", "llamafactory.cli", "train", job.config_path or ""]
    return [runner["cli_path"], "train", job.config_path or ""]


def _run_preflight_checks(db: Session, job: TrainingJob, runner: dict[str, Any] | None) -> None:
    errors: list[str] = []
    warnings: list[str] = []

    dataset = db.get(Dataset, job.dataset_id)
    if not dataset:
        errors.append("Dataset not found.")
    elif dataset.status != "converted" or not dataset.converted_file_path:
        errors.append("Dataset must be converted before training.")
    elif not Path(dataset.converted_file_path).exists():
        errors.append(f"Converted dataset file does not exist: {dataset.converted_file_path}")

    for label, value in {
        "config_path": job.config_path,
        "output_dir": job.output_dir,
        "log_path": job.log_path,
    }.items():
        if not value:
            errors.append(f"{label} is empty.")

    if job.config_path and not Path(job.config_path).exists():
        errors.append(f"Training config does not exist: {job.config_path}")
    if job.output_dir:
        Path(job.output_dir).mkdir(parents=True, exist_ok=True)
    if job.log_path:
        Path(job.log_path).parent.mkdir(parents=True, exist_ok=True)

    if runner:
        local_model_path = _local_model_path(job.base_model)
        if local_model_path and not local_model_path.exists():
            errors.append(f"Local base_model path does not exist: {local_model_path}")
        version = _runner_version(runner)
        if version:
            warnings.append(f"Runner version: {version}")
        gpu_summary = _gpu_summary()
        if gpu_summary:
            warnings.append(f"GPU detected: {gpu_summary}")
        else:
            warnings.append("No NVIDIA GPU detected through nvidia-smi; real training may fail or run slowly.")

    for warning in warnings:
        append_log(job.log_path, f"[preflight] {warning}")

    if errors:
        job.status = "failed"
        job.progress = 0
        job.error_message = " ".join(errors)
        job.updated_at = datetime.utcnow()
        db.commit()
        for error in errors:
            append_log(job.log_path, f"[preflight][ERROR] {error}")
        raise HTTPException(status_code=400, detail=job.error_message)


def _env_with_pythonpath(path: str) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = path if not existing else f"{path}{os.pathsep}{existing}"
    return env


def _runner_version(runner: dict[str, Any]) -> str | None:
    if runner["kind"] in {"rwkv_peft_vendor", "llamafactory_vendor"}:
        return runner["label"]
    return _command_output([runner["cli_path"], "--version"])


def _build_rwkv_peft_command(job: TrainingJob) -> list[str]:
    config = yaml.safe_load(Path(job.config_path or "").read_text(encoding="utf-8")) or {}
    command = [
        sys.executable,
        "train.py",
        "--load_model",
        job.base_model,
        "--proj_dir",
        str(job.output_dir),
        "--data_file",
        str(config["dataset_path"]),
        "--data_type",
        "sft",
        "--sft_field",
        "query",
        "response",
        "--sft_split",
        "train",
        "--my_testing",
        str(config.get("my_testing", "x070")),
        "--ctx_len",
        str(config.get("cutoff_len", 1024)),
        "--epoch_count",
        str(config.get("num_train_epochs", 1)),
        "--epoch_steps",
        str(config.get("epoch_steps", 100)),
        "--epoch_save",
        "1",
        "--micro_bsz",
        str(config.get("micro_bsz", 1)),
        "--accumulate_grad_batches",
        str(config.get("gradient_accumulation_steps", 1)),
        "--lr_init",
        str(config.get("learning_rate", 2e-4)),
        "--lr_final",
        str(config.get("lr_final", 1e-5)),
        "--n_layer",
        str(config.get("n_layer", 24)),
        "--n_embd",
        str(config.get("n_embd", 2048)),
        "--op",
        str(config.get("op", "cuda")),
        "--strategy",
        str(config.get("strategy", "deepspeed_stage_1")),
        "--precision",
        str(config.get("precision", "bf16")),
        "--devices",
        "1",
    ]
    if config.get("peft"):
        command.extend(["--peft", str(config["peft"])])
    if config.get("lora_config"):
        command.extend(["--lora_config", json.dumps(config["lora_config"])])
    if config.get("quant"):
        command.extend(["--quant", str(config["quant"])])
    return command


def _local_model_path(base_model: str) -> Path | None:
    expanded = os.path.expanduser(base_model)
    if expanded.startswith(("/", "./", "../", "~")):
        return Path(expanded)
    path = Path(expanded)
    if path.exists():
        return path
    return None


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = result.stdout.strip().splitlines()
    return output[0][:300] if output else None


def _gpu_summary() -> str | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return "; ".join(lines) if lines else None


def _explain_training_failure(log_path: str | None, fallback: str) -> str:
    log_tail = read_log(log_path, max_bytes=40_000).lower()
    for needle, message in FAILURE_HINTS:
        if needle in log_tail:
            return f"{fallback}. {message}"
    return fallback
