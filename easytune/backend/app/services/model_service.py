import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.database import STORAGE_ROOT
from app.models import ModelVersion, TrainingJob


CHECKSUM_MAX_BYTES = 256 * 1024 * 1024
ADAPTER_ARTIFACT_NAMES = {
    "adapter_config.json",
    "adapter_model.bin",
    "adapter_model.safetensors",
    "pytorch_model.bin",
    "model.safetensors",
}


def create_model_version_for_job(db: Session, job: TrainingJob) -> ModelVersion:
    existing = (
        db.query(ModelVersion)
        .filter(ModelVersion.training_job_id == job.id)
        .order_by(ModelVersion.id.desc())
        .first()
    )
    if existing:
        return existing

    manifest = _build_export_manifest(job)
    export_path = manifest["export_path"]
    status = "ready" if manifest["has_adapter_artifact"] else "incomplete"
    model_version = ModelVersion(
        training_job_id=job.id,
        name=f"{job.name} v1",
        base_model=job.base_model,
        adapter_path=job.output_dir or "",
        export_path=export_path,
        status=status,
    )
    db.add(model_version)
    db.commit()
    db.refresh(model_version)
    manifest["model_version_id"] = model_version.id
    _write_export_manifest(manifest)
    _register_model_version(model_version, manifest)
    return model_version


def _build_export_manifest(job: TrainingJob) -> dict[str, Any]:
    output_dir = Path(job.output_dir or "").resolve()
    export_dir = output_dir / "export"
    files = _scan_artifacts(output_dir)
    total_size = sum(item["size_bytes"] for item in files)
    has_adapter_artifact = any(item["name"] in ADAPTER_ARTIFACT_NAMES for item in files)
    return {
        "manifest_version": 1,
        "model_version_id": None,
        "training_job_id": job.id,
        "training_job_name": job.name,
        "model_family": job.model_family,
        "base_model": job.base_model,
        "adapter_path": str(output_dir),
        "export_path": str(export_dir),
        "created_at": datetime.utcnow().isoformat(),
        "artifact_count": len(files),
        "total_size_bytes": total_size,
        "has_adapter_artifact": has_adapter_artifact,
        "files": files,
    }


def _scan_artifacts(output_dir: Path) -> list[dict[str, Any]]:
    if not output_dir.exists():
        return []

    artifacts: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output_dir)
        if relative.parts and relative.parts[0] == "export":
            continue
        stat = path.stat()
        artifacts.append(
            {
                "path": str(relative),
                "name": path.name,
                "size_bytes": stat.st_size,
                "modified_at": datetime.utcfromtimestamp(stat.st_mtime).isoformat(),
                "sha256": _sha256_if_reasonable(path, stat.st_size),
            }
        )
    return artifacts


def _sha256_if_reasonable(path: Path, size_bytes: int) -> str | None:
    if size_bytes > CHECKSUM_MAX_BYTES:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_export_manifest(manifest: dict[str, Any]) -> None:
    export_dir = Path(manifest["export_path"])
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (export_dir / "README.txt").write_text(
        "EasyTune export manifest. Adapter files remain in adapter_path; this directory records the validated artifact list.\n",
        encoding="utf-8",
    )


def _register_model_version(model_version: ModelVersion, manifest: dict[str, Any]) -> None:
    registry_dir = STORAGE_ROOT / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    registry_path = registry_dir / "model_versions.json"
    records: list[dict[str, Any]] = []
    if registry_path.exists():
        try:
            loaded = json.loads(registry_path.read_text(encoding="utf-8") or "[]")
            if isinstance(loaded, list):
                records = loaded
        except json.JSONDecodeError:
            records = []

    records = [record for record in records if record.get("model_version_id") != model_version.id]
    records.append(
        {
            "model_version_id": model_version.id,
            "training_job_id": model_version.training_job_id,
            "name": model_version.name,
            "base_model": model_version.base_model,
            "adapter_path": model_version.adapter_path,
            "export_path": model_version.export_path,
            "status": model_version.status,
            "created_at": model_version.created_at.isoformat(),
            "artifact_count": manifest["artifact_count"],
            "total_size_bytes": manifest["total_size_bytes"],
            "has_adapter_artifact": manifest["has_adapter_artifact"],
        }
    )
    registry_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
