import importlib.util
import json
import shutil
import os
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter

from app.database import APP_DIR, STORAGE_ROOT

router = APIRouter(prefix="/api/health", tags=["health"])

RWKV_PEFT_MODULES = ["torch", "lightning", "peft", "transformers", "datasets", "jsonlines", "rwkv", "numpy"]
RWKV_PEFT_OPTIONAL_MODULES = ["deepspeed", "rwkvfla", "bitsandbytes", "wandb"]
QWEN_MODULES = ["torch", "transformers", "datasets", "accelerate"]
RWKV_ML_VENV_PYTHON = APP_DIR.parent / ".venv-ml" / "bin" / "python"


def gpu_summary() -> list[dict[str, str]]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        name, memory_used, memory_total, utilization = parts
        gpus.append(
            {
                "name": name,
                "memory_used_mb": memory_used,
                "memory_total_mb": memory_total,
                "utilization_percent": utilization,
            }
        )
    return gpus


def dependency_summary(required: list[str], optional: list[str] | None = None) -> dict[str, object]:
    optional = optional or []
    required_status = {module: importlib.util.find_spec(module) is not None for module in required}
    optional_status = {module: importlib.util.find_spec(module) is not None for module in optional}
    return {
        "available": all(required_status.values()),
        "required": required_status,
        "optional": optional_status,
        "missing": [module for module, installed in required_status.items() if not installed],
        "missing_optional": [module for module, installed in optional_status.items() if not installed],
    }


def rwkv_python_executable() -> str:
    configured = os.getenv("EASYTUNE_RWKV_PYTHON")
    if configured:
        return configured
    if RWKV_ML_VENV_PYTHON.exists():
        return str(RWKV_ML_VENV_PYTHON)
    return sys.executable


def python_dependency_summary(
    python_executable: str,
    required: list[str],
    optional: list[str] | None = None,
) -> dict[str, object]:
    optional = optional or []
    modules = required + optional
    if not Path(python_executable).exists():
        required_status = {module: False for module in required}
        optional_status = {module: False for module in optional}
    else:
        script = (
            "import importlib.util, json, sys; "
            "mods=json.loads(sys.argv[1]); "
            "print(json.dumps({m: importlib.util.find_spec(m) is not None for m in mods}))"
        )
        try:
            result = subprocess.run(
                [python_executable, "-c", script, json.dumps(modules)],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            )
            status = json.loads(result.stdout or "{}") if result.returncode == 0 else {}
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            status = {}
        required_status = {module: bool(status.get(module)) for module in required}
        optional_status = {module: bool(status.get(module)) for module in optional}
    return {
        "python": python_executable,
        "available": all(required_status.values()),
        "required": required_status,
        "optional": optional_status,
        "missing": [module for module, installed in required_status.items() if not installed],
        "missing_optional": [module for module, installed in optional_status.items() if not installed],
    }


@router.get("")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "storage_root": str(STORAGE_ROOT),
        "llamafactory_cli_available": shutil.which("llamafactory-cli") is not None,
        "rwkv_finetune_available": shutil.which("rwkv-finetune") is not None,
        "inference_provider": os.getenv("EASYTUNE_INFERENCE_PROVIDER", "disabled"),
        "inference_base_url": os.getenv("EASYTUNE_INFERENCE_BASE_URL"),
        "api_token_required": bool(os.getenv("EASYTUNE_API_TOKEN")),
        "cors_origins": os.getenv("EASYTUNE_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"),
        "gpus": gpu_summary(),
        "vendor": {
            "rwkv_lightning": (APP_DIR / "vendor" / "rwkv_lightning" / "app.py").exists(),
            "rwkv_peft": (APP_DIR / "vendor" / "rwkv_peft" / "train.py").exists(),
            "llamafactory": (APP_DIR / "vendor" / "llamafactory" / "src" / "llamafactory").exists(),
        },
        "dependencies": {
            "rwkv_peft": python_dependency_summary(rwkv_python_executable(), RWKV_PEFT_MODULES, RWKV_PEFT_OPTIONAL_MODULES),
            "qwen_training": dependency_summary(QWEN_MODULES),
        },
    }
