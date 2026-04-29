import shutil
import json
import os
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
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "save_steps": 100,
    },
    "balanced": {
        "num_train_epochs": 3,
        "lora_rank": 16,
        "cutoff_len": 4096,
        "learning_rate": 2.0e-4,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "save_steps": 100,
    },
    "high": {
        "num_train_epochs": 5,
        "lora_rank": 32,
        "cutoff_len": 8192,
        "learning_rate": 1.0e-4,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "save_steps": 200,
    },
}


def _has_nvidia_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None


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
        rwkv_dataset_path = _write_rwkv_peft_sft_dataset(dataset)
        config = {
            "model_family": "rwkv",
            "runner": "rwkv_finetune",
            "model_name_or_path": job.base_model,
            "stage": job.stage,
            "do_train": True,
            "finetuning_type": job.finetuning_type,
            "template": "rwkv",
            "dataset": dataset_name(dataset.id),
            "dataset_path": str(rwkv_dataset_path.resolve()),
            "dataset_dir": str((STORAGE_ROOT / "llamafactory_data").resolve()),
            "cutoff_len": preset["cutoff_len"],
            "learning_rate": preset["learning_rate"],
            "lr_final": float(os.getenv("EASYTUNE_RWKV_LR_FINAL", "1e-5")),
            "num_train_epochs": preset["num_train_epochs"],
            "epoch_steps": int(os.getenv("EASYTUNE_RWKV_EPOCH_STEPS", "100")),
            "micro_bsz": preset["per_device_train_batch_size"],
            "gradient_accumulation_steps": preset["gradient_accumulation_steps"],
            "lr_scheduler_type": "cosine",
            "logging_steps": 10,
            "save_steps": preset["save_steps"],
            "output_dir": str(output_dir),
            "overwrite_output_dir": True,
            "plot_loss": True,
            "bf16": _has_nvidia_gpu(),
            "my_testing": os.getenv("EASYTUNE_RWKV_MY_TESTING", "x070"),
            "n_layer": int(os.getenv("EASYTUNE_RWKV_N_LAYER", "24")),
            "n_embd": int(os.getenv("EASYTUNE_RWKV_N_EMBD", "2048")),
            "op": os.getenv("EASYTUNE_RWKV_OP", "cuda"),
            "strategy": os.getenv("EASYTUNE_RWKV_STRATEGY", "deepspeed_stage_1"),
            "precision": os.getenv("EASYTUNE_RWKV_PRECISION", "bf16"),
        }
        if job.finetuning_type in {"lora", "qlora"}:
            config["peft"] = "lora"
            config["lora_config"] = {
                "lora_load": "",
                "lora_r": preset["lora_rank"],
                "lora_alpha": preset["lora_rank"] * 2,
                "lora_dropout": 0.01,
            }
        if job.finetuning_type == "qlora":
            config["quant"] = os.getenv("EASYTUNE_RWKV_QUANT", "nf4")
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
        "per_device_train_batch_size": preset["per_device_train_batch_size"],
        "gradient_accumulation_steps": preset["gradient_accumulation_steps"],
        "lr_scheduler_type": "cosine",
        "logging_steps": 10,
        "save_steps": preset["save_steps"],
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "plot_loss": True,
        "bf16": _has_nvidia_gpu(),
        "gradient_checkpointing": True,
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "trust_remote_code": True,
        "report_to": "none",
    }

    if job.finetuning_type in {"lora", "qlora"}:
        config["lora_rank"] = preset["lora_rank"]
        config["lora_target"] = "all"
    if job.finetuning_type == "qlora":
        config["quantization_bit"] = 4
        config["quantization_method"] = "bitsandbytes"

    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    log_path.touch(exist_ok=True)
    return {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "command": f"llamafactory-cli train {config_path}",
    }


def _write_rwkv_peft_sft_dataset(dataset: Dataset) -> Path:
    source_path = Path(dataset.converted_file_path or "")
    target_path = STORAGE_ROOT / "llamafactory_data" / f"{dataset_name(dataset.id)}_rwkv_sft.jsonl"
    rows = []
    for line in source_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        query, response = _record_to_query_response(record)
        if query and response:
            rows.append({"query": query, "response": response})

    with target_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target_path


def _record_to_query_response(record: dict) -> tuple[str, str]:
    if "instruction" in record or "output" in record:
        instruction = str(record.get("instruction", "")).strip()
        input_text = str(record.get("input", "")).strip()
        response = str(record.get("output", "")).strip()
        query = "\n".join(part for part in [instruction, input_text] if part).strip()
        return query, response

    messages = record.get("messages") or record.get("conversations") or []
    if not isinstance(messages, list):
        return "", ""
    last_user = ""
    last_assistant = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or message.get("from") or "").lower()
        content = str(message.get("content") or message.get("value") or "").strip()
        if role in {"user", "human"}:
            last_user = content
        if role in {"assistant", "gpt"}:
            last_assistant = content
    return last_user, last_assistant
