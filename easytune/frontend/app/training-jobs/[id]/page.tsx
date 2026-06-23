"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import { TrainingJob, apiFetch, errorMessage } from "../../../lib/api";
import styles from "../../shared.module.scss";
import { statusClass } from "../../ui";

type LogsResponse = {
  job_id: number;
  log_path: string | null;
  content: string;
};

export default function TrainingJobDetailPage() {
  const params = useParams<{ id: string }>();
  const jobId = Number(params.id);
  const [job, setJob] = useState<TrainingJob | null>(null);
  const [logs, setLogs] = useState("");
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    const [jobData, logData] = await Promise.all([
      apiFetch<TrainingJob>(`/api/training-jobs/${jobId}`),
      apiFetch<LogsResponse>(`/api/training-jobs/${jobId}/logs`)
    ]);
    setJob(jobData);
    setLogs(logData.content);
  }, [jobId]);

  useEffect(() => {
    load().catch((err: unknown) => setError(errorMessage(err)));
    const timer = window.setInterval(() => {
      load().catch((err: unknown) => setError(errorMessage(err)));
    }, 2000);
    return () => window.clearInterval(timer);
  }, [load]);

  async function start() {
    setError("");
    try {
      await apiFetch<TrainingJob>(`/api/training-jobs/${jobId}/start`, { method: "POST" });
      await load();
    } catch (err) {
      setError(errorMessage(err));
    }
  }

  if (!job) {
    return <div className={styles.panel}>加载中...</div>;
  }

  return (
    <>
      <div className={styles.titleRow}>
        <div>
          <h1>{job.name}</h1>
          <p>任务详情每 2 秒刷新一次日志。</p>
        </div>
        <div className={styles.actions}>
          <Link className={styles.secondaryButton} href="/training-jobs">
            返回列表
          </Link>
          <button
            className={styles.button}
            disabled={job.status === "running" || job.status === "completed"}
            onClick={start}
            type="button"
          >
            启动
          </button>
        </div>
      </div>

      {error && <div className={styles.error}>{error}</div>}

      <section className={styles.panel}>
        <h2>任务信息</h2>
        <dl className={styles.keyValue}>
          <dt>ID</dt>
          <dd>{job.id}</dd>
          <dt>状态</dt>
          <dd>
            <span className={statusClass(job.status)}>{job.status}</span>
          </dd>
          <dt>进度</dt>
          <dd>{job.progress}%</dd>
          <dt>模型方向</dt>
          <dd>{job.model_family}</dd>
          <dt>base_model</dt>
          <dd>{job.base_model}</dd>
          <dt>template</dt>
          <dd>{job.template}</dd>
          <dt>stage</dt>
          <dd>{job.stage}</dd>
          <dt>finetuning_type</dt>
          <dd>{job.finetuning_type}</dd>
          <dt>budget_level</dt>
          <dd>{job.budget_level}</dd>
          {job.extra_config_json && (
            <>
              <dt>extra_config</dt>
              <dd>
                <pre className={styles.code}>{JSON.stringify(job.extra_config_json, null, 2)}</pre>
              </dd>
            </>
          )}
          <dt>command</dt>
          <dd>{job.command}</dd>
          <dt>config_path</dt>
          <dd>{job.config_path}</dd>
          <dt>log_path</dt>
          <dd>{job.log_path}</dd>
          <dt>output_dir</dt>
          <dd>{job.output_dir}</dd>
          <dt>error_message</dt>
          <dd>{job.error_message || "-"}</dd>
        </dl>
      </section>

      <section className={styles.panel}>
        <h2>实时日志</h2>
        <pre className={styles.code}>{logs || "暂无日志"}</pre>
      </section>
    </>
  );
}
