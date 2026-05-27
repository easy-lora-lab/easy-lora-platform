"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import { ModelVersion, apiFetch, errorMessage } from "../../../lib/api";
import styles from "../../shared.module.scss";
import { formatDate } from "../../ui";

export default function ModelVersionDetailPage() {
  const params = useParams<{ id: string }>();
  const [version, setVersion] = useState<ModelVersion | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    apiFetch<ModelVersion>(`/api/model-versions/${Number(params.id)}`)
      .then((item) => setVersion(item))
      .catch((err: unknown) => setError(errorMessage(err)));
  }, [params.id]);

  if (!version) {
    return <div className={styles.panel}>加载中...</div>;
  }

  return (
    <>
      <div className={styles.titleRow}>
        <div>
          <h1>{version.name}</h1>
          <p>模型版本详情。</p>
        </div>
        <div className={styles.actions}>
          <Link className={styles.secondaryButton} href="/model-versions">
            返回列表
          </Link>
          <Link className={styles.button} href="/validation">
            人工验收
          </Link>
        </div>
      </div>

      {error && <div className={styles.error}>{error}</div>}

      <section className={styles.panel}>
        <h2>基础信息</h2>
        <dl className={styles.keyValue}>
          <dt>ID</dt>
          <dd>{version.id}</dd>
          <dt>training_job_id</dt>
          <dd>{version.training_job_id}</dd>
          <dt>base_model</dt>
          <dd>{version.base_model}</dd>
          <dt>adapter_path</dt>
          <dd>{version.adapter_path}</dd>
          <dt>export_path</dt>
          <dd>{version.export_path || "-"}</dd>
          <dt>状态</dt>
          <dd>{version.status}</dd>
          <dt>创建时间</dt>
          <dd>{formatDate(version.created_at)}</dd>
        </dl>
      </section>
    </>
  );
}
