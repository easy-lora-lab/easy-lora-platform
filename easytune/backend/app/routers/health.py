import shutil

from fastapi import APIRouter

from app.database import STORAGE_ROOT

router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "storage_root": str(STORAGE_ROOT),
        "llamafactory_cli_available": shutil.which("llamafactory-cli") is not None,
    }
