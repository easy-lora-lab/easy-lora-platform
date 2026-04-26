"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { ModelVersion, apiFetch } from "../../lib/api";
import styles from "../shared.module.scss";

function statusClass(status: string) {
  if (status === "ready") return `${styles.status} ${styles.statusReady}`;
  if (status === "failed") return `${styles.status} ${styles.statusFailed}`;
  return styles.status;
}

export default function ModelVersionsPage() {
  const [versions, setVersions] = useState<ModelVersion[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    apiFetch<ModelVersion[]>("/api/model-versions")
      .then((items) => setVersions(items))
      .catch((err: Error) => setError(err.message));
  }, []);

  return (
    <>
      <div className={styles.titleRow}>
        <div>
          <h1>模型版本管理</h1>
          <p>训练完成后自动生成版本记录，导出路径暂作预留。</p>
        </div>
      </div>

      {error && <div className={styles.error}>{error}</div>}

      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>ID</th>
              <th>模型名称</th>
              <th>训练任务</th>
              <th>base_model</th>
              <th>adapter_path</th>
              <th>export_path</th>
              <th>状态</th>
              <th>创建时间</th>
            </tr>
          </thead>
          <tbody>
            {versions.map((version) => (
              <tr key={version.id}>
                <td>{version.id}</td>
                <td>
                  <Link href={`/model-versions/${version.id}`}>{version.name}</Link>
                </td>
                <td>
                  <Link href={`/training-jobs/${version.training_job_id}`}>{version.training_job_id}</Link>
                </td>
                <td>{version.base_model}</td>
                <td>{version.adapter_path}</td>
                <td>{version.export_path || "-"}</td>
                <td>
                  <span className={statusClass(version.status)}>{version.status}</span>
                </td>
                <td>{new Date(version.created_at).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
