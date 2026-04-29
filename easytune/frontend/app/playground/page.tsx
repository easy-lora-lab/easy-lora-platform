"use client";

import { FormEvent, useEffect, useState } from "react";

import { ModelVersion, ValidationGenerateResponse, apiFetch } from "../../lib/api";
import styles from "../shared.module.scss";

export default function PlaygroundPage() {
  const [versions, setVersions] = useState<ModelVersion[]>([]);
  const [modelVersionId, setModelVersionId] = useState("");
  const [prompt, setPrompt] = useState("");
  const [fineTunedAnswer, setFineTunedAnswer] = useState("");
  const [baseAnswer, setBaseAnswer] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    apiFetch<ModelVersion[]>("/api/model-versions")
      .then((items) => setVersions(items))
      .catch((err: Error) => setError(err.message));
  }, []);

  async function run(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setFineTunedAnswer("");
    try {
      const response = await apiFetch<ValidationGenerateResponse>("/api/validation-records/generate", {
        method: "POST",
        body: JSON.stringify({
          model_version_id: Number(modelVersionId),
          prompt,
          temperature: 0.2,
          max_tokens: 512
        })
      });
      setFineTunedAnswer(response.actual_answer);
      setBaseAnswer("Base 模型对比需要在 Deploy / Inference 中配置第二个推理后端后接入。");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <div className={styles.titleRow}>
        <div>
          <span className={styles.eyebrow}>Chat Playground</span>
          <h1>模型问答对比</h1>
          <p>加载本机后端配置的 RWKV Lightning、Qwen Transformers、vLLM 或 Ollama 推理服务。</p>
        </div>
      </div>

      {error && <div className={styles.error}>{error}</div>}

      <section className={styles.panel}>
        <form className={styles.form} onSubmit={run}>
          <div className={styles.field}>
            <label htmlFor="modelVersion">fine-tuned model</label>
            <select id="modelVersion" required value={modelVersionId} onChange={(event) => setModelVersionId(event.target.value)}>
              <option value="">选择模型版本</option>
              {versions.map((version) => (
                <option key={version.id} value={version.id}>
                  #{version.id} {version.name}
                </option>
              ))}
            </select>
          </div>
          <div className={styles.field}>
            <label htmlFor="prompt">prompt</label>
            <textarea id="prompt" required value={prompt} onChange={(event) => setPrompt(event.target.value)} />
          </div>
          <button className={styles.button} disabled={loading} type="submit">
            {loading ? "生成中..." : "运行对比"}
          </button>
        </form>
      </section>

      <section className={styles.compareGrid}>
        <div className={styles.panel}>
          <h2>Base</h2>
          <pre className={styles.code}>{baseAnswer || "等待运行"}</pre>
        </div>
        <div className={styles.panel}>
          <h2>Fine-tuned</h2>
          <pre className={styles.code}>{fineTunedAnswer || "等待运行"}</pre>
        </div>
      </section>
    </>
  );
}
