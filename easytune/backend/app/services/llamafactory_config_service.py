from pathlib import Path

import yaml
from fastapi import HTTPException

from app.database import STORAGE_ROOT
from app.models import Dataset, TrainingJob
from app.services.dataset_service import dataset_name


BUDGET_PRESETS = {
    "low": {
        "num_train_epochs": 1,
        "lora_rank": 8,
        "cutoff_len": 2048,
        "learning_rate": 2.0e-4,
    },
    "balanced": {
        "num_train_epochs": 3,
        "lora_rank": 16,
        "cutoff_len": 4096,
        "learning_rate": 2.0e-4,
    },
    "high": {
        "num_train_epochs": 5,
        "lora_rank": 32,
        "cutoff_len": 8192,
        "learning_rate": 1.0e-4,
    },
}


def generate_training_yaml(job: TrainingJob, dataset: Dataset) -> dict[str, str]:
    if dataset.status != "converted" or not dataset.converted_file_path:
        raise HTTPException(status_code=400, detail="Dataset must be converted before creating a training job.")
    if job.budget_level not in BUDGET_PRESETS:
        raise HTTPException(status_code=400, detail="budget_level must be low, balanced or high.")

    preset = BUDGET_PRESETS[job.budget_level]
    output_dir = (STORAGE_ROOT / "outputs" / f"job_{job.id}").resolve()
    log_path = (STORAGE_ROOT / "logs" / f"job_{job.id}.log").resolve()
    config_path = (STORAGE_ROOT / "configs" / f"job_{job.id}.yaml").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if job.model_family == "rwkv":
        config = {
            "model_family": "rwkv",
            "runner": "rwkv_placeholder",
            "model_name_or_path": job.base_model,
            "stage": job.stage,
            "do_train": True,
            "finetuning_type": job.finetuning_type,
            "template": "rwkv",
            "dataset": dataset_name(dataset.id),
            "dataset_dir": str((STORAGE_ROOT / "llamafactory_data").resolve()),
            "cutoff_len": preset["cutoff_len"],
            "learning_rate": preset["learning_rate"],
            "num_train_epochs": preset["num_train_epochs"],
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "lr_scheduler_type": "cosine",
            "logging_steps": 10,
            "save_steps": 100,
            "output_dir": str(output_dir),
            "overwrite_output_dir": True,
            "plot_loss": True,
            "bf16": True,
            "placeholder_note": "RWKV finetuning runner is reserved for intern integration.",
        }
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        log_path.touch(exist_ok=True)
        return {
            "config_path": str(config_path),
            "output_dir": str(output_dir),
            "log_path": str(log_path),
            "command": f"rwkv-finetune train {config_path}",
        }

    config = {
        "model_family": "qwen",
        "model_name_or_path": job.base_model,
        "stage": job.stage,
        "do_train": True,
        "finetuning_type": job.finetuning_type,
        "template": job.template,
        "dataset": dataset_name(dataset.id),
        "dataset_dir": str((STORAGE_ROOT / "llamafactory_data").resolve()),
        "cutoff_len": preset["cutoff_len"],
        "learning_rate": preset["learning_rate"],
        "num_train_epochs": preset["num_train_epochs"],
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "lr_scheduler_type": "cosine",
        "logging_steps": 10,
        "save_steps": 100,
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "plot_loss": True,
        "bf16": True,
    }

    if job.finetuning_type == "lora":
        config["lora_rank"] = preset["lora_rank"]
        config["lora_target"] = "all"

    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    log_path.touch(exist_ok=True)
    return {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "command": f"llamafactory-cli train {config_path}",
    }
