"use client";

import { FormEvent, useEffect, useState } from "react";

import { Health, apiFetch, errorMessage, getApiBaseUrl, getApiToken, setApiBaseUrl, setApiToken } from "../../lib/api";
import styles from "../shared.module.scss";

export default function DeployPage() {
  const [apiBase, setApiBase] = useState("");
  const [apiToken, setApiTokenValue] = useState("");
  const [health, setHealth] = useState<Health | null>(null);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    setApiBase(getApiBaseUrl());
    setApiTokenValue(getApiToken());
    apiFetch<Health>("/api/health")
      .then((item) => setHealth(item))
      .catch((err: unknown) => setError(errorMessage(err)));
  }, []);

  function saveApiBase(event: FormEvent) {
    event.preventDefault();
    setApiBaseUrl(apiBase);
    setApiToken(apiToken);
    setNotice("前端 API 地址和访问令牌已保存到当前浏览器。");
  }

  return (
    <>
      <div className={styles.titleRow}>
        <div>
          <span className={styles.eyebrow}>Deploy / Inference</span>
          <h1>前后端分离部署</h1>
          <p>前端可部署在阿里云，浏览器端通过这里配置本机后端地址；后端留在本机负责 GPU、训练和推理。</p>
        </div>
      </div>

      {notice && <div className={styles.notice}>{notice}</div>}
      {error && <div className={styles.error}>{error}</div>}

      <section className={styles.deployGrid}>
        <div className={styles.panel}>
          <h2>前端 API 地址</h2>
          <form className={styles.form} onSubmit={saveApiBase}>
            <div className={styles.field}>
              <label htmlFor="apiBase">Backend API Base URL</label>
              <input id="apiBase" value={apiBase} onChange={(event) => setApiBase(event.target.value)} />
            </div>
            <div className={styles.field}>
              <label htmlFor="apiToken">API Token</label>
              <input
                id="apiToken"
                type="password"
                value={apiToken}
                onChange={(event) => setApiTokenValue(event.target.value)}
                placeholder="后端设置 EASYTUNE_API_TOKEN 时填写"
              />
            </div>
            <button className={styles.button} type="submit">
              保存
            </button>
          </form>
        </div>

        <div className={styles.panel}>
          <h2>后端健康状态</h2>
          <div className={styles.configList}>
            <div>
              <span>status</span>
              <strong>{health?.status || "-"}</strong>
            </div>
            <div>
              <span>storage</span>
              <strong>{health?.storage_root || "-"}</strong>
            </div>
            <div>
              <span>Qwen train</span>
              <strong>{health?.llamafactory_cli_available ? "system cli" : "vendor LLaMA-Factory / mock"}</strong>
            </div>
            <div>
              <span>RWKV train</span>
              <strong>{health?.rwkv_finetune_available ? "system cli" : "vendor RWKV-PEFT / mock"}</strong>
            </div>
            <div>
              <span>inference</span>
              <strong>{health?.inference_provider || "disabled"}</strong>
            </div>
            <div>
              <span>api token</span>
              <strong>{health?.api_token_required ? "required" : "not required"}</strong>
            </div>
            <div>
              <span>cors</span>
              <strong>{health?.cors_origins || "-"}</strong>
            </div>
          </div>
        </div>
      </section>

      <section className={styles.panel}>
        <h2>本机后端推荐启动方式</h2>
        <pre className={styles.code}>{`cd easytune/backend
source .venv/bin/activate
export EASYTUNE_INFERENCE_PROVIDER=rwkv_lightning
export EASYTUNE_INFERENCE_BASE_URL=http://127.0.0.1:8001
export EASYTUNE_API_TOKEN=change-this-token
export EASYTUNE_CORS_ORIGINS=http://localhost:3000
uvicorn app.main:app --host 0.0.0.0 --port 8000

# RWKV Lightning 内置推理服务
cd easytune/backend/app/vendor/rwkv_lightning
python app.py --model-path /path/to/rwkv-model.pth --port 8001 --password optional-token`}</pre>
      </section>
    </>
  );
}
