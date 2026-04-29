import json
import os
import urllib.error
import urllib.request
from typing import Any

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models import ModelVersion
from app.schemas import ValidationGenerateRequest, ValidationGenerateResponse


OPENAI_COMPATIBLE_PROVIDERS = {"openai_compatible", "openai-compatible", "vllm", "openwebui", "open_webui"}
_QWEN_CACHE: dict[str, Any] = {}


def generate_validation_answer(db: Session, payload: ValidationGenerateRequest) -> ValidationGenerateResponse:
    model_version = db.get(ModelVersion, payload.model_version_id)
    if not model_version:
        raise HTTPException(status_code=404, detail="Model version not found.")

    provider = os.getenv("EASYTUNE_INFERENCE_PROVIDER", "disabled").strip().lower()
    if provider in {"", "disabled", "manual"}:
        raise HTTPException(
            status_code=400,
            detail=(
                "Inference provider is not configured. Set EASYTUNE_INFERENCE_PROVIDER to "
                "rwkv_lightning, qwen_transformers, openai_compatible, vllm, openwebui or ollama, then set "
                "EASYTUNE_INFERENCE_BASE_URL/MODEL."
            ),
        )
    if provider in {"rwkv_lightning", "rwkv-lightning", "rwkv"}:
        return _call_rwkv_lightning(model_version, payload)
    if provider in {"qwen_transformers", "transformers", "local_qwen"}:
        return _call_qwen_transformers(model_version, payload)
    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        return _call_openai_compatible(provider, model_version, payload)
    if provider == "ollama":
        return _call_ollama(model_version, payload)
    raise HTTPException(status_code=400, detail=f"Unsupported inference provider: {provider}")


def _call_qwen_transformers(
    model_version: ModelVersion,
    payload: ValidationGenerateRequest,
) -> ValidationGenerateResponse:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="qwen_transformers requires torch and transformers in the backend Python environment.",
        ) from exc

    base_model = os.getenv("EASYTUNE_QWEN_BASE_MODEL") or model_version.base_model
    adapter_path = os.getenv("EASYTUNE_QWEN_ADAPTER_PATH") or model_version.adapter_path
    cache_key = f"{base_model}|{adapter_path}"
    if cache_key not in _QWEN_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=os.getenv("EASYTUNE_QWEN_DEVICE_MAP", "auto"),
            torch_dtype=getattr(torch, os.getenv("EASYTUNE_QWEN_TORCH_DTYPE", "bfloat16")),
            trust_remote_code=True,
        )
        if adapter_path and os.path.exists(adapter_path):
            try:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, adapter_path)
            except ImportError as exc:
                raise HTTPException(
                    status_code=500,
                    detail="Loading a Qwen adapter requires peft in the backend Python environment.",
                ) from exc
        model.eval()
        _QWEN_CACHE[cache_key] = (tokenizer, model)

    tokenizer, model = _QWEN_CACHE[cache_key]
    messages = []
    if payload.system_prompt:
        messages.append({"role": "system", "content": payload.system_prompt})
    messages.append({"role": "user", "content": payload.prompt})
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = "\n".join(f"{item['role']}: {item['content']}" for item in messages) + "\nassistant:"
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=payload.max_tokens,
            temperature=payload.temperature,
            do_sample=payload.temperature > 0,
        )
    generated_ids = output_ids[:, inputs.input_ids.shape[-1] :]
    answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return ValidationGenerateResponse(
        model_version_id=model_version.id,
        provider="qwen_transformers",
        model=base_model,
        actual_answer=answer,
        raw_response=None,
    )


def _call_rwkv_lightning(
    model_version: ModelVersion,
    payload: ValidationGenerateRequest,
) -> ValidationGenerateResponse:
    base_url = os.getenv("EASYTUNE_INFERENCE_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
    model = _served_model_name(model_version, default="rwkv7")
    messages = []
    if payload.system_prompt:
        messages.append({"role": "system", "content": payload.system_prompt})
    messages.append({"role": "user", "content": payload.prompt})
    body = {
        "model": model,
        "messages": messages,
        "temperature": payload.temperature,
        "max_tokens": payload.max_tokens,
        "stream": False,
        "top_k": int(os.getenv("EASYTUNE_RWKV_TOP_K", "20")),
        "top_p": float(os.getenv("EASYTUNE_RWKV_TOP_P", "0.6")),
        "alpha_presence": float(os.getenv("EASYTUNE_RWKV_ALPHA_PRESENCE", "1")),
        "alpha_frequency": float(os.getenv("EASYTUNE_RWKV_ALPHA_FREQUENCY", "0.1")),
        "alpha_decay": float(os.getenv("EASYTUNE_RWKV_ALPHA_DECAY", "0.996")),
        "enable_think": os.getenv("EASYTUNE_RWKV_ENABLE_THINK", "false").lower() == "true",
        "use_prefix_cache": os.getenv("EASYTUNE_RWKV_USE_PREFIX_CACHE", "true").lower() != "false",
    }
    password = os.getenv("EASYTUNE_INFERENCE_API_KEY") or os.getenv("EASYTUNE_RWKV_LIGHTNING_PASSWORD")
    if password:
        body["password"] = password
    headers = {"Authorization": f"Bearer {password}"} if password else {}
    response = _post_json(f"{base_url}/openai/v1/chat/completions", body, headers=headers)
    content = _extract_openai_content(response)
    return ValidationGenerateResponse(
        model_version_id=model_version.id,
        provider="rwkv_lightning",
        model=str(response.get("model") or model),
        actual_answer=content,
        raw_response=response,
    )


def _call_openai_compatible(
    provider: str,
    model_version: ModelVersion,
    payload: ValidationGenerateRequest,
) -> ValidationGenerateResponse:
    base_url = os.getenv("EASYTUNE_INFERENCE_BASE_URL", "").rstrip("/")
    if not base_url:
        raise HTTPException(status_code=400, detail="EASYTUNE_INFERENCE_BASE_URL is required for OpenAI-compatible inference.")
    model = _served_model_name(model_version)
    messages = []
    if payload.system_prompt:
        messages.append({"role": "system", "content": payload.system_prompt})
    messages.append({"role": "user", "content": payload.prompt})
    body = {
        "model": model,
        "messages": messages,
        "temperature": payload.temperature,
        "max_tokens": payload.max_tokens,
    }
    headers = {}
    api_key = os.getenv("EASYTUNE_INFERENCE_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = _post_json(f"{base_url}/chat/completions", body, headers=headers)
    content = _extract_openai_content(response)
    return ValidationGenerateResponse(
        model_version_id=model_version.id,
        provider=provider,
        model=model,
        actual_answer=content,
        raw_response=response,
    )


def _call_ollama(model_version: ModelVersion, payload: ValidationGenerateRequest) -> ValidationGenerateResponse:
    base_url = os.getenv("EASYTUNE_INFERENCE_BASE_URL", "http://localhost:11434").rstrip("/")
    model = _served_model_name(model_version)
    prompt = payload.prompt
    if payload.system_prompt:
        prompt = f"{payload.system_prompt}\n\n{payload.prompt}"
    response = _post_json(
        f"{base_url}/api/generate",
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": payload.temperature,
                "num_predict": payload.max_tokens,
            },
        },
        headers={},
    )
    content = response.get("response")
    if not isinstance(content, str) or not content.strip():
        raise HTTPException(status_code=502, detail="Ollama response has no content.")
    return ValidationGenerateResponse(
        model_version_id=model_version.id,
        provider="ollama",
        model=model,
        actual_answer=content.strip(),
        raw_response=response,
    )


def _served_model_name(model_version: ModelVersion, default: str | None = None) -> str:
    return os.getenv("EASYTUNE_INFERENCE_MODEL") or default or model_version.name or model_version.base_model


def _extract_openai_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise HTTPException(status_code=502, detail="Inference response has no choices.")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise HTTPException(status_code=502, detail="Inference response has no message content.")
    return content.strip()


def _post_json(url: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    timeout = float(os.getenv("EASYTUNE_INFERENCE_TIMEOUT", "60"))
    request = urllib.request.Request(
        url=url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=502, detail=f"Inference server returned {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"Inference server is unreachable: {exc.reason}") from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Inference request timed out.") from exc

    try:
        loaded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="Inference server returned non-JSON response.") from exc
    if not isinstance(loaded, dict):
        raise HTTPException(status_code=502, detail="Inference server returned an unexpected response shape.")
    return loaded
