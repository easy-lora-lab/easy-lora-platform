"use client";

import { useEffect, useState } from "react";

import { ModelVersion, ValidationRecord, apiFetch } from "../../lib/api";
import styles from "../shared.module.scss";

export default function EvaluationPage() {
  const [versions, setVersions] = useState<ModelVersion[]>([]);
  const [records, setRecords] = useState<ValidationRecord[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([apiFetch<ModelVersion[]>("/api/model-versions"), apiFetch<ValidationRecord[]>("/api/validation-records")])
      .then(([versionData, recordData]) => {
        setVersions(versionData);
        setRecords(recordData);
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  return (
    <>
      <div className={styles.titleRow}>
        <div>
          <span className={styles.eyebrow}>Evaluation</span>
          <h1>评测与对比</h1>
          <p>集中管理 benchmark、自定义测试集和 base vs fine-tuned 对比结果。</p>
        </div>
      </div>

      {error && <div className={styles.error}>{error}</div>}

      <section className={styles.metricGrid}>
        <div className={styles.metricCard}>
          <span>模型版本</span>
          <strong>{versions.length}</strong>
          <small>可参与评测的 adapter/checkpoint</small>
        </div>
        <div className={styles.metricCard}>
          <span>人工记录</span>
          <strong>{records.length}</strong>
          <small>来自人工验收与 Playground</small>
        </div>
        <div className={styles.metricCard}>
          <span>Benchmark</span>
          <strong>Ready</strong>
          <small>接入自定义 JSONL 测试集</small>
        </div>
      </section>

      <section className={styles.compareGrid}>
        <div className={styles.panel}>
          <h2>自动 benchmark</h2>
          <div className={styles.configList}>
            <div>
              <span>输入</span>
              <strong>JSONL / CSV 测试集</strong>
            </div>
            <div>
              <span>指标</span>
              <strong>准确率、人工评分、失败样本</strong>
            </div>
            <div>
              <span>状态</span>
              <strong>后端评测 runner 待接 API</strong>
            </div>
          </div>
        </div>
        <div className={styles.panel}>
          <h2>base vs fine-tuned</h2>
          <p className={styles.muted}>当前已有模型版本和人工验收记录，下一步可以把 Playground 的双模型输出保存为评测记录。</p>
          <pre className={styles.code}>{JSON.stringify(records.slice(0, 5), null, 2)}</pre>
        </div>
      </section>
    </>
  );
}
