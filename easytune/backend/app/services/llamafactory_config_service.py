import shutil
import json
import os
from pathlib import Path
from typing import Any

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

RWKV_DEFAULT_MODEL_SHAPES = {
    "x070": {"n_layer": 24, "n_embd": 2048, "vocab_size": 65536},
    "x060": {"n_layer": 24, "n_embd": 2048, "vocab_size": 65536},
    "x052": {"n_layer": 24, "n_embd": 2048, "vocab_size": 65536},
}


def _has_nvidia_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None


def generate_training_yaml(job: TrainingJob, dataset: Dataset) -> dict[str, str]:
    if dataset.status != "converted" or not dataset.converted_file_path:
        raise HTTPException(status_code=400, detail="Dataset must be converted before creating a training job.")
    if job.budget_level not in BUDGET_PRESETS:
        raise HTTPException(status_code=400, detail="budget_level must be low, balanced or high.")

    preset = BUDGET_PRESETS[job.budget_level]
    config_path, output_dir, log_path = _training_paths(job.id)

    if job.model_family == "rwkv":
        extra = _extra_config(job)
        rwkv_data_type = _string_value(extra, "data_type", "sft").lower()
        if rwkv_data_type not in {"sft", "jsonl"}:
            raise HTTPException(status_code=400, detail="RWKV data_type must be sft or jsonl.")
        if rwkv_data_type == "jsonl":
            rwkv_dataset_path = _write_rwkv_peft_text_dataset(dataset)
        else:
            rwkv_dataset_path = _write_rwkv_peft_sft_dataset(dataset)
        config = _rwkv_training_config(job, dataset, preset, output_dir, rwkv_dataset_path, rwkv_data_type)
        return _write_config_response(config, config_path, output_dir, log_path, f"rwkv-finetune train {config_path}")

    config = {
        "model_family": "qwen",
        **_base_training_config(job, dataset, preset, output_dir),
        "template": job.template,
        "per_device_train_batch_size": preset["per_device_train_batch_size"],
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

    return _write_config_response(config, config_path, output_dir, log_path, f"llamafactory-cli train {config_path}")


def _training_paths(job_id: int) -> tuple[Path, Path, Path]:
    output_dir = (STORAGE_ROOT / "outputs" / f"job_{job_id}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return (
        (STORAGE_ROOT / "configs" / f"job_{job_id}.yaml").resolve(),
        output_dir,
        (STORAGE_ROOT / "logs" / f"job_{job_id}.log").resolve(),
    )


def _base_training_config(job: TrainingJob, dataset: Dataset, preset: dict, output_dir: Path) -> dict:
    return {
        "model_name_or_path": job.base_model,
        "stage": job.stage,
        "do_train": True,
        "finetuning_type": job.finetuning_type,
        "dataset": dataset_name(dataset.id),
        "dataset_dir": str((STORAGE_ROOT / "llamafactory_data").resolve()),
        "cutoff_len": preset["cutoff_len"],
        "learning_rate": preset["learning_rate"],
        "num_train_epochs": preset["num_train_epochs"],
        "gradient_accumulation_steps": preset["gradient_accumulation_steps"],
        "lr_scheduler_type": "cosine",
        "logging_steps": 10,
        "save_steps": preset["save_steps"],
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "plot_loss": True,
        "bf16": _has_nvidia_gpu(),
    }


def _rwkv_training_config(
    job: TrainingJob,
    dataset: Dataset,
    preset: dict[str, Any],
    output_dir: Path,
    rwkv_dataset_path: Path,
    data_type: str,
) -> dict[str, Any]:
    extra = _extra_config(job)
    my_testing = _string_value(extra, "my_testing", os.getenv("EASYTUNE_RWKV_MY_TESTING", "x070"))
    model_shape = RWKV_DEFAULT_MODEL_SHAPES.get(my_testing, RWKV_DEFAULT_MODEL_SHAPES["x070"])
    peft_method = _rwkv_peft_method(job.finetuning_type)
    peft = _rwkv_runner_peft_name(peft_method)
    quant_default = os.getenv("EASYTUNE_RWKV_QUANT", "nf4" if peft_method == "qlora" else "none")
    op_default = os.getenv("EASYTUNE_RWKV_OP", "fla" if peft_method == "state" else "cuda")
    train_type_default = "state" if peft_method == "state" else "none"
    ctx_len = _int_value(extra, "ctx_len", int(os.getenv("EASYTUNE_RWKV_CTX_LEN", preset["cutoff_len"])))

    config: dict[str, Any] = {
        "model_family": "rwkv",
        **_base_training_config(job, dataset, preset, output_dir),
        "runner": "rwkv_finetune",
        "template": "rwkv",
        "dataset_path": str(rwkv_dataset_path.resolve()),
        "data_type": data_type,
        "sft_field": ["query", "response"],
        "sft_split": _string_value(extra, "sft_split", "train"),
        "my_testing": my_testing,
        "vocab_size": _int_value(extra, "vocab_size", int(model_shape["vocab_size"])),
        "ctx_len": ctx_len,
        "chunk_ctx": _int_value(extra, "chunk_ctx", min(512, ctx_len)),
        "epoch_count": _int_value(extra, "epoch_count", preset["num_train_epochs"]),
        "epoch_begin": _int_value(extra, "epoch_begin", 0),
        "epoch_steps": _int_value(extra, "epoch_steps", int(os.getenv("EASYTUNE_RWKV_EPOCH_STEPS", "100"))),
        "epoch_save": _int_value(extra, "epoch_save", 1),
        "micro_bsz": _int_value(extra, "micro_bsz", preset["per_device_train_batch_size"]),
        "n_layer": _int_value(extra, "n_layer", int(os.getenv("EASYTUNE_RWKV_N_LAYER", model_shape["n_layer"]))),
        "n_embd": _int_value(extra, "n_embd", int(os.getenv("EASYTUNE_RWKV_N_EMBD", model_shape["n_embd"]))),
        "lr_init": _float_value(extra, "lr_init", preset["learning_rate"]),
        "lr_final": _float_value(extra, "lr_final", float(os.getenv("EASYTUNE_RWKV_LR_FINAL", "1e-5"))),
        "warmup_steps": _int_value(extra, "warmup_steps", 0),
        "beta1": _float_value(extra, "beta1", 0.9),
        "beta2": _float_value(extra, "beta2", 0.99),
        "adam_eps": _float_value(extra, "adam_eps", 1e-8),
        "grad_cp": _int_value(extra, "grad_cp", 1),
        "dropout": _float_value(extra, "dropout", 0),
        "weight_decay": _float_value(extra, "weight_decay", 0),
        "layerwise_lr": _int_value(extra, "layerwise_lr", 1),
        "ds_bucket_mb": _int_value(extra, "ds_bucket_mb", 200),
        "head_size_a": _int_value(extra, "head_size_a", 64),
        "head_size_divisor": _int_value(extra, "head_size_divisor", 8),
        "op": _string_value(extra, "op", op_default),
        "fused_kernel": _bool_value(extra, "fused_kernel", False),
        "train_type": _string_value(extra, "train_type", train_type_default),
        "dataload": _string_value(extra, "dataload", "pad"),
        "loss_mask": _string_value(extra, "loss_mask", "none"),
        "data_shuffle": _int_value(extra, "data_shuffle", 1),
        "lr_schedule": _string_value(extra, "lr_schedule", "cos"),
        "optimizer": _string_value(extra, "optimizer", os.getenv("EASYTUNE_RWKV_OPTIMIZER", "none")),
        "peft": peft,
        "peft_method": peft_method,
        "quant": _string_value(extra, "quant", quant_default),
        "train_parts": _list_value(extra.get("train_parts"), ["time", "ln"]),
        "accelerator": _string_value(extra, "accelerator", "gpu"),
        "strategy": _string_value(extra, "strategy", os.getenv("EASYTUNE_RWKV_STRATEGY", "deepspeed_stage_1")),
        "devices": _int_value(extra, "devices", 1),
        "num_nodes": _int_value(extra, "num_nodes", 1),
        "precision": _string_value(extra, "precision", os.getenv("EASYTUNE_RWKV_PRECISION", "bf16")),
        "accumulate_grad_batches": _int_value(extra, "accumulate_grad_batches", preset["gradient_accumulation_steps"]),
        "num_workers": _int_value(extra, "num_workers", 8),
        "persistent_workers": _bool_value(extra, "persistent_workers", False),
    }

    peft_config = _rwkv_peft_config(peft_method, extra, preset)
    if peft_config:
        config["peft_config"] = peft_config
    if config["quant"] == "none":
        config.pop("quant")
    return config


def _extra_config(job: TrainingJob) -> dict[str, Any]:
    return job.extra_config_json if isinstance(job.extra_config_json, dict) else {}


def _rwkv_peft_method(finetuning_type: str) -> str:
    method = finetuning_type.strip().lower()
    if method == "none":
        return "full"
    return method


def _rwkv_runner_peft_name(peft_method: str) -> str:
    if peft_method in {"full", "none"}:
        return "none"
    if peft_method == "qlora":
        return "lora"
    if peft_method == "disha":
        return "miss"
    return peft_method


def _rwkv_peft_config(peft_method: str, extra: dict[str, Any], preset: dict[str, Any]) -> dict[str, Any] | None:
    user_config = extra.get("peft_config")
    if isinstance(user_config, dict):
        return user_config
    if peft_method in {"full", "none", "state"}:
        return None

    rank = _int_value(extra, "peft_rank", preset["lora_rank"])
    if peft_method in {"lora", "qlora", "adalora"}:
        return {
            "r": rank,
            "lora_alpha": _int_value(extra, "lora_alpha", rank * 2),
            "lora_dropout": _float_value(extra, "lora_dropout", 0.01),
        }
    if peft_method in {"miss", "disha"}:
        return {"r": rank}
    if peft_method == "prefix":
        return {"num_virtual_tokens": _int_value(extra, "num_virtual_tokens", 16)}
    return {"r": rank}


def _string_value(extra: dict[str, Any], key: str, default: str) -> str:
    value = extra.get(key, default)
    if value is None or value == "":
        return str(default)
    return str(value)


def _int_value(extra: dict[str, Any], key: str, default: int) -> int:
    value = extra.get(key, default)
    if value is None or value == "":
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{key} must be an integer.") from exc


def _float_value(extra: dict[str, Any], key: str, default: float) -> float:
    value = extra.get(key, default)
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{key} must be a number.") from exc


def _bool_value(extra: dict[str, Any], key: str, default: bool) -> bool:
    value = extra.get(key, default)
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _list_value(value: Any, default: list[str]) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [item.strip() for item in value.split(",") if item.strip()]
    return default


def _write_config_response(
    config: dict,
    config_path: Path,
    output_dir: Path,
    log_path: Path,
    command: str,
) -> dict[str, str]:
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    log_path.touch(exist_ok=True)
    return {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "command": command,
    }


def _write_rwkv_peft_sft_dataset(dataset: Dataset) -> Path:
    source_path = Path(dataset.converted_file_path or "")
    target_path = STORAGE_ROOT / "llamafactory_data" / f"{dataset_name(dataset.id)}_rwkv_sft.jsonl"
    written = 0
    with target_path.open("w", encoding="utf-8") as handle:
        for line in source_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            query, response = _record_to_query_response(json.loads(line))
            if query and response:
                handle.write(json.dumps({"query": query, "response": response}, ensure_ascii=False) + "\n")
                written += 1
    if written == 0:
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Converted dataset has no usable RWKV SFT query/response pairs.")
    return target_path


def _write_rwkv_peft_text_dataset(dataset: Dataset) -> Path:
    source_path = Path(dataset.converted_file_path or "")
    target_path = STORAGE_ROOT / "llamafactory_data" / f"{dataset_name(dataset.id)}_rwkv_text.jsonl"
    written = 0
    with target_path.open("w", encoding="utf-8") as handle:
        for line in source_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            query, response = _record_to_query_response(json.loads(line))
            if query and response:
                text = f"User: {query}\nAssistant: {response}"
                handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                written += 1
    if written == 0:
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Converted dataset has no usable RWKV text records.")
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
