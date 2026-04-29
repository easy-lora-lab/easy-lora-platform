"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import { Dataset, ModelVersion, TrainingJob, apiFetch } from "../lib/api";
import styles from "./shared.module.scss";

type Health = {
  status: string;
  llamafactory_cli_available: boolean;
  rwkv_finetune_available: boolean;
  inference_provider: string;
  inference_base_url: string | null;
  gpus: Array<{
    name: string;
    memory_used_mb: string;
    memory_total_mb: string;
    utilization_percent: string;
  }>;
};

function statusClass(status: string) {
  if (status === "completed" || status === "ready") return `${styles.status} ${styles.statusReady}`;
  if (status === "running") return `${styles.status} ${styles.statusRunning}`;
  if (status === "failed" || status === "incomplete") return `${styles.status} ${styles.statusFailed}`;
  return styles.status;
}

export default function DashboardPage() {
  const [health, setHealth] = useState<Health | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [versions, setVersions] = useState<ModelVersion[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([
      apiFetch<Health>("/api/health"),
      apiFetch<Dataset[]>("/api/datasets"),
      apiFetch<TrainingJob[]>("/api/training-jobs"),
      apiFetch<ModelVersion[]>("/api/model-versions")
    ])
      .then(([healthData, datasetData, jobData, versionData]) => {
        setHealth(healthData);
        setDatasets(datasetData);
        setJobs(jobData);
        setVersions(versionData);
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  const failedJobs = useMemo(() => jobs.filter((job) => job.status === "failed"), [jobs]);
  const runningJobs = useMemo(() => jobs.filter((job) => job.status === "running"), [jobs]);
  const recentJobs = jobs.slice(0, 5);

  return (
    <>
      <section className={styles.heroPanel}>
        <div>
          <span className={styles.eyebrow}>Dashboard</span>
          <h1>训练与推理总览</h1>
          <p>关注 GPU、最近任务、失败任务、模型版本和数据资产，入口按微调平台的真实工作流组织。</p>
        </div>
        <div className={styles.heroStats}>
          <div>
            <span>Qwen Runner</span>
            <strong>{health?.llamafactory_cli_available ? "Ready" : "Mock"}</strong>
          </div>
          <div>
            <span>RWKV Runner</span>
            <strong>{health?.rwkv_finetune_available ? "Ready" : "Mock"}</strong>
          </div>
          <div>
            <span>Inference</span>
            <strong>{health?.inference_provider || "disabled"}</strong>
          </div>
        </div>
      </section>

      {error && <div className={styles.error}>{error}</div>}

      <section className={styles.metricGrid}>
        <div className={styles.metricCard}>
          <span>GPU 状态</span>
          <strong>{health?.gpus.length ? `${health.gpus.length} cards` : "未检测到"}</strong>
          <small>{health?.gpus[0]?.name || "nvidia-smi 不可用或没有 GPU"}</small>
        </div>
        <div className={styles.metricCard}>
          <span>运行中任务</span>
          <strong>{runningJobs.length}</strong>
          <small>Training Monitor 实时查看日志和进度</small>
        </div>
        <div className={styles.metricCard}>
          <span>失败任务</span>
          <strong>{failedJobs.length}</strong>
          <small>启动前置检查与失败摘要会写入日志</small>
        </div>
        <div className={styles.metricCard}>
          <span>模型版本</span>
          <strong>{versions.length}</strong>
          <small>adapter / export / registry</small>
        </div>
        <div className={styles.metricCard}>
          <span>数据集</span>
          <strong>{datasets.length}</strong>
          <small>上传、质检、转换与 split</small>
        </div>
      </section>

      <section className={styles.sectionGrid}>
        <div className={styles.panel}>
          <div className={styles.panelHeader}>
            <div>
              <h2>最近训练任务</h2>
              <p className={styles.muted}>按创建时间倒序显示，点击进入监控页。</p>
            </div>
            <Link className={styles.secondaryButton} href="/training-jobs">
              查看全部
            </Link>
          </div>
          <div className={styles.timeline}>
            {recentJobs.map((job) => (
              <Link href={`/training-jobs/${job.id}`} key={job.id}>
                <span className={statusClass(job.status)}>{job.status}</span>
                <strong>{job.name}</strong>
                <small>
                  {job.model_family} / {job.progress}%
                </small>
              </Link>
            ))}
            {!recentJobs.length && <p className={styles.muted}>暂无训练任务。</p>}
          </div>
        </div>

        <div className={styles.panel}>
          <div className={styles.panelHeader}>
            <div>
              <h2>GPU 显存</h2>
              <p className={styles.muted}>来自后端健康检查的 nvidia-smi 摘要。</p>
            </div>
          </div>
          <div className={styles.gpuList}>
            {health?.gpus.map((gpu) => {
              const used = Number(gpu.memory_used_mb);
              const total = Number(gpu.memory_total_mb);
              const ratio = total > 0 ? Math.min(100, Math.round((used / total) * 100)) : 0;
              return (
                <div key={gpu.name}>
                  <div>
                    <strong>{gpu.name}</strong>
                    <span>{gpu.utilization_percent}% util</span>
                  </div>
                  <div className={styles.progressTrack}>
                    <span style={{ width: `${ratio}%` }} />
                  </div>
                  <small>
                    {gpu.memory_used_mb} / {gpu.memory_total_mb} MB
                  </small>
                </div>
              );
            })}
            {!health?.gpus.length && <p className={styles.muted}>当前环境没有可展示的 GPU 数据。</p>}
          </div>
        </div>
      </section>
    </>
  );
}
