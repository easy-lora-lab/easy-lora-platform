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
RWKV_WEIGHT_SUFFIXES = {".pth"}


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
    adapter_path = manifest["adapter_path"]
    status = "ready" if manifest["has_model_artifact"] else "incomplete"
    model_version = ModelVersion(
        training_job_id=job.id,
        name=f"{job.name} v1",
        base_model=job.base_model,
        adapter_path=adapter_path,
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
    scan_roots = _artifact_scan_roots(output_dir)
    files = _scan_artifacts(scan_roots)
    adapter_path = _select_model_artifact_path(output_dir, files)
    total_size = sum(item["size_bytes"] for item in files)
    has_adapter_artifact = any(item["name"] in ADAPTER_ARTIFACT_NAMES for item in files)
    has_rwkv_weight_artifact = any(_is_rwkv_weight_artifact(item) for item in files)
    has_model_artifact = has_adapter_artifact or has_rwkv_weight_artifact
    return {
        "manifest_version": 1,
        "model_version_id": None,
        "training_job_id": job.id,
        "training_job_name": job.name,
        "model_family": job.model_family,
        "base_model": job.base_model,
        "adapter_path": str(adapter_path or output_dir),
        "export_path": str(export_dir),
        "created_at": datetime.utcnow().isoformat(),
        "artifact_count": len(files),
        "total_size_bytes": total_size,
        "has_adapter_artifact": has_adapter_artifact,
        "has_rwkv_weight_artifact": has_rwkv_weight_artifact,
        "has_model_artifact": has_model_artifact,
        "scan_roots": [str(root) for root in scan_roots],
        "files": files,
    }


def _artifact_scan_roots(output_dir: Path) -> list[Path]:
    roots = [output_dir]
    sibling_adapter_dir = output_dir.parent / f"{output_dir.name}-adapter"
    if sibling_adapter_dir.exists():
        roots.append(sibling_adapter_dir.resolve())
    return roots


def _scan_artifacts(roots: list[Path]) -> list[dict[str, Any]]:
    existing_roots = [root for root in roots if root.exists()]
    if not existing_roots:
        return []

    artifacts: list[dict[str, Any]] = []
    for root in existing_roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if relative.parts and relative.parts[0] == "export":
                continue
            stat = path.stat()
            artifacts.append(
                {
                    "root": str(root),
                    "path": str(relative),
                    "absolute_path": str(path),
                    "name": path.name,
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.utcfromtimestamp(stat.st_mtime).isoformat(),
                    "sha256": _sha256_if_reasonable(path, stat.st_size),
                }
            )
    return artifacts


def _select_model_artifact_path(output_dir: Path, files: list[dict[str, Any]]) -> Path | None:
    adapter_dirs = sorted(
        {
            str(Path(item["absolute_path"]).parent)
            for item in files
            if item["name"] in ADAPTER_ARTIFACT_NAMES
        }
    )
    if adapter_dirs:
        return Path(adapter_dirs[-1])

    rwkv_weights = sorted(
        (Path(item["absolute_path"]) for item in files if _is_rwkv_weight_artifact(item)),
        key=lambda path: path.stat().st_mtime,
    )
    if rwkv_weights:
        return rwkv_weights[-1]
    return output_dir if output_dir.exists() else None


def _is_rwkv_weight_artifact(item: dict[str, Any]) -> bool:
    path = Path(str(item["name"]))
    return path.suffix.lower() in RWKV_WEIGHT_SUFFIXES and path.name.startswith("rwkv-")


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
            "has_rwkv_weight_artifact": manifest["has_rwkv_weight_artifact"],
            "has_model_artifact": manifest["has_model_artifact"],
        }
    )
    registry_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
