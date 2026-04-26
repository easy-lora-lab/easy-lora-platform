"use client";

import Link from "next/link";
import { FormEvent, useEffect, useState } from "react";

import { Dataset, apiFetch, uploadDataset } from "../../lib/api";
import styles from "../shared.module.scss";

function statusClass(status: string) {
  if (status === "converted") return `${styles.status} ${styles.statusReady}`;
  if (status === "failed") return `${styles.status} ${styles.statusFailed}`;
  return styles.status;
}

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function loadDatasets() {
    setDatasets(await apiFetch<Dataset[]>("/api/datasets"));
  }

  useEffect(() => {
    loadDatasets().catch((err: Error) => setError(err.message));
  }, []);

  async function handleUpload(event: FormEvent) {
    event.preventDefault();
    if (!file) {
      setError("请选择数据集文件。");
      return;
    }
    setLoading(true);
    setError("");
    setNotice("");
    try {
      const created = await uploadDataset(file, name);
      setNotice(`已上传数据集：${created.name}`);
      setName("");
      setFile(null);
      await loadDatasets();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  async function handleConvert(datasetId: number) {
    setLoading(true);
    setError("");
    setNotice("");
    try {
      await apiFetch<Dataset>(`/api/datasets/${datasetId}/convert`, { method: "POST" });
      setNotice(`数据集 ${datasetId} 已转换。`);
      await loadDatasets();
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
          <h1>数据集管理</h1>
          <p>上传数据集后自动质检，转换后生成 LLaMA-Factory 数据文件和 dataset_info.json。</p>
        </div>
      </div>

      {notice && <div className={styles.notice}>{notice}</div>}
      {error && <div className={styles.error}>{error}</div>}

      <section className={styles.panel}>
        <h2>上传数据集</h2>
        <form className={styles.form} onSubmit={handleUpload}>
          <div className={styles.inlineFields}>
            <div className={styles.field}>
              <label htmlFor="name">数据集名称</label>
              <input id="name" value={name} onChange={(event) => setName(event.target.value)} placeholder="可选" />
            </div>
            <div className={styles.field}>
              <label htmlFor="file">文件</label>
              <input
                id="file"
                type="file"
                accept=".json,.jsonl,.csv"
                onChange={(event) => setFile(event.target.files?.[0] || null)}
              />
            </div>
          </div>
          <div>
            <button className={styles.button} disabled={loading} type="submit">
              上传并质检
            </button>
          </div>
        </form>
      </section>

      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>ID</th>
              <th>名称</th>
              <th>格式</th>
              <th>样本数</th>
              <th>质量分</th>
              <th>状态</th>
              <th>路径</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            {datasets.map((dataset) => (
              <tr key={dataset.id}>
                <td>{dataset.id}</td>
                <td>
                  <Link href={`/datasets/${dataset.id}`}>{dataset.name}</Link>
                </td>
                <td>{dataset.formatting}</td>
                <td>{dataset.sample_count}</td>
                <td>{dataset.quality_score}</td>
                <td>
                  <span className={statusClass(dataset.status)}>{dataset.status}</span>
                </td>
                <td>{dataset.converted_file_path || dataset.original_file_path}</td>
                <td>
                  <div className={styles.actions}>
                    <Link className={styles.secondaryButton} href={`/datasets/${dataset.id}`}>
                      详情
                    </Link>
                    <button
                      className={styles.button}
                      disabled={loading || dataset.status === "converted" || dataset.formatting === "unknown"}
                      onClick={() => handleConvert(dataset.id)}
                      type="button"
                    >
                      转换
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
