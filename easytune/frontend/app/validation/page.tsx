"use client";

import { FormEvent, useEffect, useState } from "react";

import { ModelVersion, ValidationRecord, apiFetch } from "../../lib/api";
import styles from "../shared.module.scss";

export default function ValidationPage() {
  const [versions, setVersions] = useState<ModelVersion[]>([]);
  const [modelVersionId, setModelVersionId] = useState("");
  const [prompt, setPrompt] = useState("");
  const [expectedAnswer, setExpectedAnswer] = useState("");
  const [actualAnswer, setActualAnswer] = useState("");
  const [humanScore, setHumanScore] = useState("3");
  const [humanNote, setHumanNote] = useState("");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    apiFetch<ModelVersion[]>("/api/model-versions")
      .then((items) => setVersions(items))
      .catch((err: Error) => setError(err.message));
  }, []);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    setNotice("");
    try {
      await apiFetch<ValidationRecord>("/api/validation-records", {
        method: "POST",
        body: JSON.stringify({
          model_version_id: Number(modelVersionId),
          prompt,
          expected_answer: expectedAnswer || null,
          actual_answer: actualAnswer,
          human_score: Number(humanScore),
          human_note: humanNote || null
        })
      });
      setNotice("验收记录已保存。");
      setPrompt("");
      setExpectedAnswer("");
      setActualAnswer("");
      setHumanNote("");
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <>
      <div className={styles.titleRow}>
        <div>
          <h1>人工验收</h1>
          <p>第一版不做真实推理，只保存人工填写的验收记录。</p>
        </div>
      </div>

      {notice && <div className={styles.notice}>{notice}</div>}
      {error && <div className={styles.error}>{error}</div>}

      <section className={styles.panel}>
        <form className={styles.form} onSubmit={submit}>
          <div className={styles.field}>
            <label htmlFor="modelVersion">模型版本</label>
            <select
              id="modelVersion"
              required
              value={modelVersionId}
              onChange={(event) => setModelVersionId(event.target.value)}
            >
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

          <div className={styles.field}>
            <label htmlFor="expectedAnswer">expected_answer</label>
            <textarea
              id="expectedAnswer"
              value={expectedAnswer}
              onChange={(event) => setExpectedAnswer(event.target.value)}
            />
          </div>

          <div className={styles.field}>
            <label htmlFor="actualAnswer">actual_answer</label>
            <textarea
              id="actualAnswer"
              required
              value={actualAnswer}
              onChange={(event) => setActualAnswer(event.target.value)}
            />
          </div>

          <div className={styles.inlineFields}>
            <div className={styles.field}>
              <label htmlFor="humanScore">human_score</label>
              <select id="humanScore" value={humanScore} onChange={(event) => setHumanScore(event.target.value)}>
                <option value="1">1</option>
                <option value="2">2</option>
                <option value="3">3</option>
                <option value="4">4</option>
                <option value="5">5</option>
              </select>
            </div>
            <div className={styles.field}>
              <label htmlFor="humanNote">human_note</label>
              <input id="humanNote" value={humanNote} onChange={(event) => setHumanNote(event.target.value)} />
            </div>
          </div>

          <button className={styles.button} type="submit">
            保存验收记录
          </button>
        </form>
      </section>
    </>
  );
}
