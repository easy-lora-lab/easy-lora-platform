import csv
import json
import random
import re
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.database import STORAGE_ROOT
from app.models import Dataset


ALLOWED_TYPES = {"json", "jsonl", "csv"}


def dataset_name(dataset_id: int) -> str:
    return f"dataset_{dataset_id}"


def _safe_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
    return cleaned or "dataset"


async def create_dataset_from_upload(db: Session, upload: UploadFile, name: str | None = None) -> Dataset:
    suffix = Path(upload.filename or "").suffix.lower().lstrip(".")
    if suffix not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Only json, jsonl and csv files are supported.")

    content = await upload.read()
    safe_name = _safe_filename(upload.filename or f"dataset.{suffix}")
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    target_path = STORAGE_ROOT / "datasets" / f"{timestamp}_{safe_name}"
    target_path.write_bytes(content)

    report = analyze_file(target_path, suffix)
    dataset = Dataset(
        name=name or Path(safe_name).stem,
        original_file_path=str(target_path),
        file_type=suffix,
        file_size=len(content),
        sample_count=report["sample_count"],
        formatting=report["detected_format"],
        quality_score=report["quality_score"],
        report_json=report,
        status="checked" if report["quality_score"] > 0 else "failed",
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    report_path = STORAGE_ROOT / "reports" / f"dataset_{dataset.id}_quality.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return dataset


def analyze_file(path: Path, file_type: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    raw_lines = text.splitlines()
    non_empty_lines = [line for line in raw_lines if line.strip()]
    empty_line_count = len(raw_lines) - len(non_empty_lines)
    line_lengths = [len(line) for line in non_empty_lines]

    errors: list[str] = []
    warnings: list[str] = []
    records: list[dict[str, Any]] = []

    try:
        records = _parse_records(text, file_type)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Parse failed: {exc}")

    sample_count = len(records)
    format_stats = _format_stats(records)
    detected_format = _detect_format(format_stats, sample_count)
    quality_score = _quality_score(
        sample_count=sample_count,
        detected_format=detected_format,
        empty_line_count=empty_line_count,
        line_count=max(len(raw_lines), 1),
        max_line_length=max(line_lengths, default=0),
        format_stats=format_stats,
        errors=errors,
        warnings=warnings,
    )

    return {
        "is_empty": path.stat().st_size == 0 or sample_count == 0,
        "line_count": len(raw_lines),
        "non_empty_line_count": len(non_empty_lines),
        "sample_count": sample_count,
        "empty_line_count": empty_line_count,
        "max_line_length": max(line_lengths, default=0),
        "avg_line_length": round(sum(line_lengths) / len(line_lengths), 2) if line_lengths else 0,
        "detected_format": detected_format,
        "alpaca_like_count": format_stats["alpaca_like_count"],
        "sharegpt_like_count": format_stats["sharegpt_like_count"],
        "openai_messages_like_count": format_stats["openai_messages_like_count"],
        "errors": errors,
        "warnings": warnings,
        "quality_score": quality_score,
    }


def _parse_records(text: str, file_type: str) -> list[dict[str, Any]]:
    if not text.strip():
        return []

    if file_type == "json":
        loaded = json.loads(text)
        if isinstance(loaded, list):
            return [item for item in loaded if isinstance(item, dict)]
        if isinstance(loaded, dict):
            return [loaded]
        return []

    if file_type == "jsonl":
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            records.append(value)
        return records

    if file_type == "csv":
        reader = csv.DictReader(StringIO(text))
        return [dict(row) for row in reader]

    raise ValueError(f"unsupported file type: {file_type}")


def load_dataset_records(dataset: Dataset) -> list[dict[str, Any]]:
    return _parse_records(Path(dataset.original_file_path).read_text(encoding="utf-8-sig", errors="replace"), dataset.file_type)


def _format_stats(records: list[dict[str, Any]]) -> dict[str, int]:
    alpaca = 0
    sharegpt = 0
    openai_messages = 0
    for record in records:
        if "instruction" in record and "output" in record:
            alpaca += 1
        conversations = record.get("conversations")
        if isinstance(conversations, list) and conversations:
            sharegpt += 1
        messages = record.get("messages")
        if isinstance(messages, list) and messages:
            openai_messages += 1
    return {
        "alpaca_like_count": alpaca,
        "sharegpt_like_count": sharegpt,
        "openai_messages_like_count": openai_messages,
    }


def _detect_format(stats: dict[str, int], sample_count: int) -> str:
    if sample_count == 0:
        return "unknown"
    threshold = max(1, sample_count // 2)
    if stats["alpaca_like_count"] >= threshold:
        return "alpaca"
    if stats["sharegpt_like_count"] >= threshold or stats["openai_messages_like_count"] >= threshold:
        return "sharegpt"
    return "unknown"


def _quality_score(
    sample_count: int,
    detected_format: str,
    empty_line_count: int,
    line_count: int,
    max_line_length: int,
    format_stats: dict[str, int],
    errors: list[str],
    warnings: list[str],
) -> float:
    if sample_count == 0:
        return 0

    score = 100
    if errors:
        score -= 80
    if detected_format == "unknown":
        score -= 40
        warnings.append("Dataset format is unknown. Conversion may fail.")

    empty_ratio = empty_line_count / line_count
    if empty_ratio > 0.1:
        score -= 20
        warnings.append("More than 10% of lines are empty.")
    elif empty_ratio > 0.03:
        score -= 10

    if max_line_length > 20000:
        score -= 15
        warnings.append("Some samples are very long.")
    elif max_line_length > 8000:
        score -= 5

    if detected_format == "alpaca":
        missing_ratio = 1 - format_stats["alpaca_like_count"] / sample_count
        score -= int(missing_ratio * 40)
    if detected_format == "sharegpt":
        like_count = max(format_stats["sharegpt_like_count"], format_stats["openai_messages_like_count"])
        missing_ratio = 1 - like_count / sample_count
        score -= int(missing_ratio * 40)

    return float(max(0, min(100, score)))


def convert_dataset(db: Session, dataset_id: int) -> Dataset:
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    if dataset.formatting not in {"alpaca", "sharegpt"}:
        raise HTTPException(status_code=400, detail="Only alpaca and sharegpt datasets can be converted.")

    records = load_dataset_records(dataset)
    if not records:
        raise HTTPException(status_code=400, detail="Dataset has no valid records.")

    normalized, info_entry = _normalize_for_llamafactory(dataset.id, dataset.formatting, records)
    target_file = STORAGE_ROOT / "llamafactory_data" / f"{dataset_name(dataset.id)}.jsonl"
    with target_file.open("w", encoding="utf-8") as handle:
        for record in normalized:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    dataset_info_path = STORAGE_ROOT / "llamafactory_data" / "dataset_info.json"
    existing_info = {}
    if dataset_info_path.exists():
        existing_info = json.loads(dataset_info_path.read_text(encoding="utf-8") or "{}")
    existing_info[dataset_name(dataset.id)] = info_entry
    dataset_info_path.write_text(json.dumps(existing_info, ensure_ascii=False, indent=2), encoding="utf-8")

    dataset.converted_file_path = str(target_file)
    dataset.dataset_info_json = {dataset_name(dataset.id): info_entry}
    dataset.status = "converted"
    db.commit()
    db.refresh(dataset)
    return dataset


def preview_dataset(db: Session, dataset_id: int, limit: int = 8) -> tuple[list[dict[str, Any]], int]:
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    records = load_dataset_records(dataset)
    return records[: max(1, min(limit, 50))], len(records)


def split_dataset(db: Session, dataset_id: int, valid_ratio: float = 0.1, seed: int = 42) -> dict[str, Any]:
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    if dataset.status != "converted" or not dataset.converted_file_path:
        raise HTTPException(status_code=400, detail="Dataset must be converted before splitting.")

    source_path = Path(dataset.converted_file_path)
    records = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) < 2:
        raise HTTPException(status_code=400, detail="Dataset needs at least 2 records for train/valid split.")

    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    valid_count = max(1, int(round(len(records) * valid_ratio)))
    valid_indices = set(indices[:valid_count])
    train_records = [record for index, record in enumerate(records) if index not in valid_indices]
    valid_records = [record for index, record in enumerate(records) if index in valid_indices]

    train_path = STORAGE_ROOT / "llamafactory_data" / f"{dataset_name(dataset.id)}_train.jsonl"
    valid_path = STORAGE_ROOT / "llamafactory_data" / f"{dataset_name(dataset.id)}_valid.jsonl"
    _write_jsonl(train_path, train_records)
    _write_jsonl(valid_path, valid_records)

    split_info = {
        "valid_ratio": valid_ratio,
        "seed": seed,
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "train_count": len(train_records),
        "valid_count": len(valid_records),
    }
    dataset.report_json = {**(dataset.report_json or {}), "split": split_info}
    db.commit()
    return {"dataset_id": dataset.id, **split_info}


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _normalize_for_llamafactory(
    dataset_id: int,
    formatting: str,
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_name = f"{dataset_name(dataset_id)}.jsonl"

    if formatting == "alpaca":
        normalized = []
        has_system = False
        has_history = False
        for record in records:
            item = {
                "instruction": str(record.get("instruction", "")),
                "input": str(record.get("input", "")),
                "output": str(record.get("output", "")),
            }
            if record.get("system"):
                item["system"] = record["system"]
                has_system = True
            if record.get("history"):
                item["history"] = record["history"]
                has_history = True
            normalized.append(item)

        columns = {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        }
        if has_system:
            columns["system"] = "system"
        if has_history:
            columns["history"] = "history"
        return normalized, {"file_name": output_name, "formatting": "alpaca", "columns": columns}

    first = records[0]
    if isinstance(first.get("conversations"), list):
        normalized = []
        has_system = False
        has_tools = False
        for record in records:
            item = {"conversations": record.get("conversations", [])}
            if record.get("system"):
                item["system"] = record["system"]
                has_system = True
            if record.get("tools"):
                item["tools"] = record["tools"]
                has_tools = True
            normalized.append(item)

        columns = {"messages": "conversations"}
        if has_system:
            columns["system"] = "system"
        if has_tools:
            columns["tools"] = "tools"
        return normalized, {"file_name": output_name, "formatting": "sharegpt", "columns": columns}

    normalized = [{"messages": record.get("messages", [])} for record in records]
    return normalized, {
        "file_name": output_name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }
