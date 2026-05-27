const DEFAULT_API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";
const API_BASE_STORAGE_KEY = "easytune_api_base_url";
const API_TOKEN_STORAGE_KEY = "easytune_api_token";

export type Dataset = {
  id: number;
  name: string;
  original_file_path: string;
  converted_file_path: string | null;
  file_type: string;
  file_size: number;
  sample_count: number;
  formatting: string;
  quality_score: number;
  report_json: Record<string, unknown> | null;
  dataset_info_json: Record<string, unknown> | null;
  status: string;
  created_at: string;
};

export type TrainingJob = {
  id: number;
  name: string;
  dataset_id: number;
  model_family: "qwen" | "rwkv";
  base_model: string;
  template: string;
  stage: string;
  finetuning_type: string;
  budget_level: string;
  config_path: string | null;
  output_dir: string | null;
  status: string;
  progress: number;
  command: string | null;
  log_path: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
};

export type ModelVersion = {
  id: number;
  training_job_id: number;
  name: string;
  base_model: string;
  adapter_path: string;
  export_path: string | null;
  status: string;
  created_at: string;
};

export type ValidationRecord = {
  id: number;
  model_version_id: number;
  prompt: string;
  expected_answer: string | null;
  actual_answer: string;
  human_score: number;
  human_note: string | null;
  created_at: string;
};

export type ValidationGenerateResponse = {
  model_version_id: number;
  provider: string;
  model: string;
  actual_answer: string;
  raw_response: Record<string, unknown> | null;
};

export type Health = {
  status: string;
  storage_root: string;
  llamafactory_cli_available: boolean;
  rwkv_finetune_available: boolean;
  inference_provider: string;
  inference_base_url: string | null;
  api_token_required?: boolean;
  cors_origins?: string;
  gpus: Array<{
    name: string;
    memory_used_mb: string;
    memory_total_mb: string;
    utilization_percent: string;
  }>;
  vendor?: Record<string, boolean>;
};

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (!(init?.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const apiToken = getApiToken();
  if (apiToken && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${apiToken}`);
  }
  const response = await fetch(`${getApiBaseUrl()}${path}`, {
    ...init,
    cache: "no-store",
    headers
  });
  if (!response.ok) {
    let message = response.statusText;
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch {
      // Ignore non-JSON error bodies.
    }
    throw new Error(message);
  }
  return response.json();
}

export async function uploadDataset(file: File, name?: string): Promise<Dataset> {
  const formData = new FormData();
  formData.append("file", file);
  if (name) {
    formData.append("name", name);
  }
  return apiFetch<Dataset>("/api/datasets/upload", {
    method: "POST",
    body: formData
  });
}

export function getApiBaseUrl(): string {
  if (typeof window === "undefined") {
    return DEFAULT_API_BASE_URL;
  }
  return window.localStorage.getItem(API_BASE_STORAGE_KEY) || DEFAULT_API_BASE_URL;
}

export function setApiBaseUrl(value: string): void {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(API_BASE_STORAGE_KEY, value.replace(/\/$/, ""));
}

export function getApiToken(): string {
  if (typeof window === "undefined") {
    return "";
  }
  return window.localStorage.getItem(API_TOKEN_STORAGE_KEY) || "";
}

export function setApiToken(value: string): void {
  if (typeof window === "undefined") {
    return;
  }
  const trimmedValue = value.trim();
  if (trimmedValue) {
    window.localStorage.setItem(API_TOKEN_STORAGE_KEY, trimmedValue);
  } else {
    window.localStorage.removeItem(API_TOKEN_STORAGE_KEY);
  }
}

export { DEFAULT_API_BASE_URL };
