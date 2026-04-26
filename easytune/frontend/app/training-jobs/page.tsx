"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { TrainingJob, apiFetch } from "../../lib/api";
import styles from "../shared.module.scss";

function statusClass(status: string) {
  if (status === "completed") return `${styles.status} ${styles.statusReady}`;
  if (status === "running") return `${styles.status} ${styles.statusRunning}`;
  if (status === "failed") return `${styles.status} ${styles.statusFailed}`;
  return styles.status;
}

export default function TrainingJobsPage() {
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  async function load() {
    setJobs(await apiFetch<TrainingJob[]>("/api/training-jobs"));
  }

  useEffect(() => {
    load().catch((err: Error) => setError(err.message));
  }, []);

  async function start(jobId: number) {
    setError("");
    setNotice("");
    try {
      await apiFetch<TrainingJob>(`/api/training-jobs/${jobId}/start`, { method: "POST" });
      setNotice(`任务 ${jobId} 已启动。`);
      await load();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <>
      <div className={styles.titleRow}>
        <div>
          <h1>训练任务</h1>
          <p>查看命令、配置、日志和输出目录，启动后自动真实训练或 mock train。</p>
        </div>
        <Link className={styles.button} href="/training-jobs/new">
          创建任务
        </Link>
      </div>

      {notice && <div className={styles.notice}>{notice}</div>}
      {error && <div className={styles.error}>{error}</div>}

      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>ID</th>
              <th>名称</th>
              <th>模型方向</th>
              <th>数据集</th>
              <th>状态</th>
              <th>进度</th>
              <th>命令</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job) => (
              <tr key={job.id}>
                <td>{job.id}</td>
                <td>
                  <Link href={`/training-jobs/${job.id}`}>{job.name}</Link>
                </td>
                <td>{job.model_family}</td>
                <td>{job.dataset_id}</td>
                <td>
                  <span className={statusClass(job.status)}>{job.status}</span>
                </td>
                <td>{job.progress}%</td>
                <td>{job.command}</td>
                <td>
                  <div className={styles.actions}>
                    <Link className={styles.secondaryButton} href={`/training-jobs/${job.id}`}>
                      详情
                    </Link>
                    <button
                      className={styles.button}
                      disabled={job.status === "running" || job.status === "completed"}
                      onClick={() => start(job.id)}
                      type="button"
                    >
                      启动
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
