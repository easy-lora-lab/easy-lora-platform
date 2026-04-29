import shutil
import os
import subprocess

from fastapi import APIRouter

from app.database import APP_DIR, STORAGE_ROOT

router = APIRouter(prefix="/api/health", tags=["health"])


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


@router.get("")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "storage_root": str(STORAGE_ROOT),
        "llamafactory_cli_available": shutil.which("llamafactory-cli") is not None,
        "rwkv_finetune_available": shutil.which("rwkv-finetune") is not None,
        "inference_provider": os.getenv("EASYTUNE_INFERENCE_PROVIDER", "disabled"),
        "inference_base_url": os.getenv("EASYTUNE_INFERENCE_BASE_URL"),
        "gpus": gpu_summary(),
        "vendor": {
            "rwkv_lightning": (APP_DIR / "vendor" / "rwkv_lightning" / "app.py").exists(),
            "rwkv_peft": (APP_DIR / "vendor" / "rwkv_peft" / "train.py").exists(),
            "llamafactory": (APP_DIR / "vendor" / "llamafactory" / "src" / "llamafactory").exists(),
        },
    }
